import os
import re
import sys
import uuid
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify, send_file, abort
from flask_socketio import SocketIO, emit
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from core.hrl_engine.hrl_agent import MacroAgent
from core.hrl_engine.micro_agent import MicroAgent
from core.generator.question_generator import QuestionGenerator
from core.evaluators.semantic_evaluator import SemanticEvaluator
from core.evaluators.audio_evaluator import AudioConfidenceEvaluator
from core.evaluators.stt_engine import SpeechToTextEngine
from core.hrl_engine.resume_parser import ResumeParser
from core.evaluators.report_manager import InterviewReportManager

from database.db_manager import DatabaseManager

# NEW: authentication module
from auth import auth_bp, login_manager, init_auth

app = Flask(__name__)
# FIX: secret key from env, never hardcoded
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", os.urandom(32).hex())

# Tighten session cookies. SESSION_COOKIE_SECURE should be True behind HTTPS.
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
if os.getenv("USE_HTTPS", "").lower() in ("1", "true", "yes"):
    app.config['SESSION_COOKIE_SECURE'] = True

# FIX: lock CORS to known origins in production. Set ALLOWED_ORIGINS in .env, comma-separated.
allowed = os.getenv("ALLOWED_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000")
allowed_origins = [o.strip() for o in allowed.split(",") if o.strip()]
socketio = SocketIO(app, cors_allowed_origins=allowed_origins, max_http_buffer_size=10_000_000)

print("\n[SYSTEM] Booting Global ML Singletons & Database...")
# These are stateless ML model wrappers — safe to share across users.
generator = QuestionGenerator()
semantic_engine = SemanticEvaluator()
audio_engine = AudioConfidenceEvaluator()
stt_engine = SpeechToTextEngine()
resume_parser = ResumeParser()
db = DatabaseManager()

# NEW: wire authentication into the app
login_manager.init_app(app)
app.register_blueprint(auth_bp)
init_auth(db)   # auth.py reuses our DatabaseManager instance, no second pool

# FIX: per-session state, keyed by SocketIO sid. No more global active_session.
sessions = {}

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Resume uploads are linked to a session token issued at upload time so concurrent
# users don't trample each other's file. The token is returned to the browser and
# echoed back in start_interview.
RESUME_TOKENS = {}  # token -> resume_path

ALLOWED_PDF_BYTES = 10 * 1024 * 1024  # 10 MB


def get_session(sid):
    """Lazy-init the per-sid state bag."""
    if sid not in sessions:
        sessions[sid] = {
            "interview_id": None,
            "round_counter": 0,
            "macro": None,
            "micro": None,
            "report_manager": InterviewReportManager(),
            "current_topic": None,
            "current_level": 1,
            "last_question": None,
            "ideal_answer": None,
            "last_candidate_answer": None,
            "in_clarification_loop": False,
            "last_was_clarification": False,
            "skills_cache": None,
            "cached_summary": None,
            "audio_path": os.path.join(UPLOAD_DIR, f"audio_{sid}.webm"),
            "resume_path": None,
            "owner_user_id": None,    # NEW: bind this session to a user
        }
    return sessions[sid]


def cleanup_session(sid):
    s = sessions.pop(sid, None)
    if not s:
        return

    # If the user uploaded a resume + started an interview but never answered a
    # single question, drop the orphan row so it doesn't litter their history.
    # Round-counter == 0 catches every flavor of abandonment (sign out, tab
    # close, network drop) and the delete itself is a no-op for any interview
    # that has rounds, so it's safe to call unconditionally.
    if s.get('interview_id') and s.get('round_counter', 0) == 0:
        try:
            db.delete_interview_if_empty(s['interview_id'])
        except Exception as e:
            print(f"[WARNING] Failed to clean up empty interview: {e}")

    for path_key in ("audio_path", "resume_path"):
        p = s.get(path_key)
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def _norm_topic(t):
    """Normalize a skill/topic name so 'React.js' and 'react js' match."""
    return re.sub(r'\s+', ' ', re.sub(r'[._\-]+', ' ', (t or '').strip().lower())).strip()


def _topic_avgs(interview_id):
    """{normalized_topic: (display_name, avg_tech_score)} for a single interview."""
    rows = db.get_interview_rounds(interview_id)
    buckets = {}
    for r in rows:
        disp = r.get('topic') or 'General'
        key = _norm_topic(disp)
        b = buckets.setdefault(key, {'disp': disp, 'scores': []})
        b['scores'].append(float(r.get('tech_score') or 0))
    return {k: (v['disp'], sum(v['scores']) / len(v['scores']))
            for k, v in buckets.items() if v['scores']}


def build_progress_context(user_id, interview_id):
    """
    Compute cross-interview progress for the PDF report — entirely in Python,
    from DB rows. The LLM never does this arithmetic, so the numbers are real.

    Returns:
      {
        'is_first': bool,                 # True when this is the user's 1st interview
        'trend':    [ {date, tech, conf}, ... ],   # oldest->newest
        'comparison': { ... } | None      # vs the immediately previous interview
      }

    Comparability rule:
      - tech score per skill is only compared for skills present in BOTH sessions
        (matched on normalized name); first-time skills are reported as a new
        baseline, never as "change".
      - vocal confidence is resume-independent and always compared.
    """
    trend_rows = db.get_user_trend_data(user_id, up_to_interview_id=interview_id)
    if not trend_rows:
        return {'is_first': True, 'trend': [], 'comparison': None}

    trend = [{
        'date': (r['session_date'].strftime('%b %d') if r.get('session_date') else '—'),
        'tech': round(float(r['avg_tech'] or 0), 1),
        'conf': round(float(r['avg_conf'] or 0), 1),
    } for r in trend_rows]

    # Find current interview in the (date-ordered) list; it's normally last.
    idx = next((i for i, r in enumerate(trend_rows) if r['id'] == interview_id),
               len(trend_rows) - 1)
    if idx <= 0:
        return {'is_first': True, 'trend': trend, 'comparison': None}

    curr, prev = trend_rows[idx], trend_rows[idx - 1]

    # Per-topic deltas for skills that recur across both sessions.
    curr_topics = _topic_avgs(curr['id'])
    prev_topics = _topic_avgs(prev['id'])
    carried, new_skills, dropped = [], [], []
    for key, (disp, avg) in curr_topics.items():
        if key in prev_topics:
            p_avg = prev_topics[key][1]
            carried.append({'topic': disp, 'prev': round(p_avg, 1),
                            'curr': round(avg, 1), 'delta': round(avg - p_avg, 1)})
        else:
            new_skills.append(disp)
    for key, (disp, _avg) in prev_topics.items():
        if key not in curr_topics:
            dropped.append(disp)
    carried.sort(key=lambda c: c['delta'], reverse=True)

    def _d(a, b):
        return round(float(a or 0) - float(b or 0), 1)

    comparison = {
        'prev_date': prev['session_date'].strftime('%b %d, %Y') if prev.get('session_date') else '—',
        'curr_date': curr['session_date'].strftime('%b %d, %Y') if curr.get('session_date') else '—',
        'tech_prev': round(float(prev['avg_tech'] or 0), 1),
        'tech_curr': round(float(curr['avg_tech'] or 0), 1),
        'tech_delta': _d(curr['avg_tech'], prev['avg_tech']),
        'conf_prev': round(float(prev['avg_conf'] or 0), 1),
        'conf_curr': round(float(curr['avg_conf'] or 0), 1),
        'conf_delta': _d(curr['avg_conf'], prev['avg_conf']),
        'carried': carried,
        'new_skills': new_skills,
        'dropped_skills': dropped,
    }
    return {'is_first': False, 'trend': trend, 'comparison': comparison}


# ============================================================
#  HTTP ROUTES
# ============================================================

@app.route('/')
@login_required
def index():
    # Pass the logged-in username into the template so the header can show it
    return render_template('index.html', username=current_user.username)


@app.route('/upload-resume', methods=['POST'])
@login_required
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400

    fname = secure_filename(file.filename)
    if not fname.lower().endswith('.pdf'):
        return jsonify({"success": False, "error": "Only PDF files are allowed."}), 400

    if request.content_length and request.content_length > ALLOWED_PDF_BYTES:
        return jsonify({"success": False, "error": "File too large (max 10 MB)."}), 400

    # Per-upload unique path so two users uploading at once don't collide
    token = uuid.uuid4().hex
    save_path = os.path.join(UPLOAD_DIR, f"resume_{token}.pdf")
    file.save(save_path)

    if os.path.getsize(save_path) > ALLOWED_PDF_BYTES:
        os.remove(save_path)
        return jsonify({"success": False, "error": "File too large (max 10 MB)."}), 400

    print(f"\n[SYSTEM] New resume uploaded by user_id={current_user.id} (token {token}).")
    try:
        skills = resume_parser.parse_skills(save_path)
    except Exception as e:
        print(f"[ERROR] Resume parse failed: {e}")
        os.remove(save_path)
        return jsonify({"success": False, "error": "Could not parse resume."}), 400

    # Tag the token with the uploading user so a different user can't claim it
    RESUME_TOKENS[token] = {
        "path": save_path,
        "skills": skills,
        "owner_user_id": current_user.id,
    }
    return jsonify({"success": True, "skills": skills, "resume_token": token})


@app.route('/download-report')
@login_required
def download_report():
    """
    Live-session download. Uses in-memory cached_summary + report_manager state
    so the PDF includes the freshest data (including determination/doubt/anxiety
    per round, which the DB also has but this path is slightly more direct).
    """
    sid = request.args.get('sid')
    if not sid or sid not in sessions:
        return "No active session found for this browser.", 404

    s = sessions[sid]

    # Ownership check — the in-memory session must belong to the requester
    if s.get('owner_user_id') != current_user.id:
        abort(403)

    if not s.get('cached_summary'):
        return "Interview not yet complete.", 400

    try:
        out_dir = os.path.join(os.path.dirname(__file__), "static", "reports")
        os.makedirs(out_dir, exist_ok=True)
        pdf_path = os.path.join(out_dir, f"report_{sid}.pdf")
        progress = build_progress_context(current_user.id, s['interview_id'])
        s['report_manager'].generate_pdf(pdf_path, s['cached_summary'], progress=progress)
        return send_file(pdf_path, as_attachment=True,
                         download_name="Executive_Interview_Report.pdf")
    except Exception as e:
        return f"An error occurred generating the report download: {str(e)}", 500


@app.route('/download-report/<int:interview_id>')
@login_required
def download_report_by_id(interview_id):
    """
    Archive download path — used by the My Interviews page to fetch any
    historical report. Reconstructs the report_manager from the DB rows
    so the PDF matches what was generated at the time of completion.
    """
    interview = db.get_interview(interview_id)
    if not interview:
        abort(404)
    if interview['user_id'] != current_user.id:
        abort(403)
    if not interview['final_report']:
        return "This interview is incomplete.", 400

    # Reconstruct a fresh report_manager from the DB rounds
    rounds = db.get_interview_rounds(interview_id)
    rm = InterviewReportManager()
    rm.set_candidate(current_user.username)   # username for the report header
    for r in rounds:
        rm.log_round(
            question=r.get('question') or '',
            transcript=r.get('transcript') or '',
            tech_score=r.get('tech_score') or 0,
            audio_analytics={
                'determination': r.get('determination') or 0.0,
                'doubt':         r.get('doubt') or 0.0,
                'anxiety':       r.get('anxiety') or 0.0,
                'confidence_score': r.get('confidence_score') or 0
            },
            topic=r.get('topic') or 'General',
            ideal_answer=r.get('ideal_answer') or '',
            rubric_reason=r.get('rubric_reason') or '',
        )

    try:
        out_dir = os.path.join(os.path.dirname(__file__), "static", "reports")
        os.makedirs(out_dir, exist_ok=True)
        pdf_path = os.path.join(out_dir, f"report_archive_{interview_id}.pdf")
        progress = build_progress_context(current_user.id, interview_id)
        rm.generate_pdf(pdf_path, interview['final_report'], progress=progress)
        return send_file(pdf_path, as_attachment=True,
                         download_name=f"Interview_Report_{interview_id}.pdf")
    except Exception as e:
        return f"An error occurred generating the archive report: {str(e)}", 500


# ============================================================
#  SocketIO
# ============================================================

@socketio.on('connect')
def handle_connect():
    # Reject unauthenticated socket connections. The session cookie comes through
    # on the WebSocket upgrade, so current_user is populated here.
    if not current_user.is_authenticated:
        print("[SOCKET] Rejected unauthenticated connection attempt.")
        return False

    sid = request.sid
    s = get_session(sid)
    s['owner_user_id'] = current_user.id   # bind the socket session to the user
    print(f"[SOCKET] Authenticated connection: sid={sid}, user={current_user.username}")
    emit('session_ready', {'sid': sid, 'username': current_user.username})


@socketio.on('disconnect')
def handle_disconnect():
    cleanup_session(request.sid)


@socketio.on('start_interview')
def handle_start_interview(data):
    if not current_user.is_authenticated:
        return
    sid = request.sid
    s = get_session(sid)

    print(f"\n[API] Initializing New Interview Session (sid={sid}, user_id={current_user.id})...")

    # NEW: bind the interview row to the logged-in user
    s['interview_id'] = db.create_interview_session(current_user.id)
    s['round_counter'] = 0
    s['report_manager'].reset()
    s['report_manager'].set_candidate(current_user.username)   # bind report to user

    # Resolve skills via token issued at upload time. Verify the token belongs
    # to this user — a token issued to user A must not be usable by user B.
    token = (data or {}).get('resume_token')
    if token and token in RESUME_TOKENS:
        entry = RESUME_TOKENS[token]
        if entry.get('owner_user_id') != current_user.id:
            emit('error', {'message': 'Resume token does not belong to this account.'})
            return
        RESUME_TOKENS.pop(token)   # consume one-shot
        s['skills_cache'] = entry['skills']
        s['resume_path'] = entry['path']
    elif s.get('skills_cache'):
        pass
    else:
        print("[WARNING] No skills cached. Using hardcoded fallbacks.")
        s['skills_cache'] = {"Python": 0.9, "SQL": 0.8}

    skills = s['skills_cache']
    # Build the rating tracker first, then hand it to the topic selector so the
    # macro agent reads real uncertainty straight from the ratings.
    s['micro'] = MicroAgent(list(skills.keys()))
    s['macro'] = MacroAgent(skills, s['micro'])

    topic = s['macro'].select_next_topic()
    level = s['micro'].get_current_level(topic)
    s['current_topic'] = topic
    s['current_level'] = level

    q_data = generator.generate_question(topic, level)
    s['last_question'] = q_data['question']
    s['ideal_answer'] = q_data['ideal_answer']
    s['macro'].register_question_asked(topic)

    emit('telemetry_update', {
        'status': 'Interview Started. Awaiting Candidate Audio.',
        'topic': topic,
        'level': level,
        'uncertainty': s['micro'].get_uncertainty(topic),
        'question': s['last_question']
    })


@socketio.on('process_live_audio')
def handle_live_audio(data):
    if not current_user.is_authenticated:
        return
    sid = request.sid
    s = get_session(sid)
    print(f"\n[API] Audio received (sid={sid}). Running Whisper STT...")
    try:
        audio_bytes = data['audio']
        with open(s['audio_path'], 'wb') as f:
            f.write(audio_bytes)
        candidate_text = stt_engine.transcribe_audio(s['audio_path'])
        emit('transcription_review', {'text': candidate_text})
    except Exception as e:
        print(f"[ERROR] STT processing failed: {e}")
        emit('error', {'message': 'Speech-to-text failed. Please try recording again.'})


@socketio.on('submit_final_answer')
def handle_final_answer(data):
    if not current_user.is_authenticated:
        return
    sid = request.sid
    s = get_session(sid)

    try:
        s['round_counter'] += 1
        topic = s['current_topic']
        level = s['current_level']
        ideal = s['ideal_answer']

        candidate_text = data.get('edited_text', '')
        s['last_candidate_answer'] = candidate_text

        print(f"\n[API] Evaluating answer for {topic} Level {level} (sid={sid})...")

        text_result = semantic_engine.evaluate_answer(ideal, candidate_text)
        audio_result = audio_engine.evaluate_audio(s['audio_path'])
        score = text_result['tech_score']
        conf_score = audio_result['confidence_score']

        # RAM log (used by the live PDF path)
        s['report_manager'].log_round(
            question=s['last_question'],
            transcript=candidate_text,
            tech_score=score,
            audio_analytics={
                'determination': audio_result.get('determination', 0),
                'doubt': audio_result.get('doubt', 0),
                'anxiety': audio_result.get('anxiety', 0),
                'confidence_score': conf_score
            },
            topic=topic,                                  # for per-topic analysis
            ideal_answer=ideal,                           # for the per-question section
            rubric_reason=text_result.get('rubric_reason', ''),  # grader text shown in UI
        )

        # Permanent MySQL log — full per-round detail so archived PDFs are
        # identical to live ones (same question, ideal, grader reason, score).
        db.log_round(
            interview_id=s['interview_id'],
            round_number=s['round_counter'],
            question=s['last_question'],
            transcript=candidate_text,
            tech_score=score,
            confidence_score=conf_score,
            determination=audio_result.get('determination', 0),
            doubt=audio_result.get('doubt', 0),
            anxiety=audio_result.get('anxiety', 0),
            topic=topic,
            ideal_answer=ideal,
            rubric_reason=text_result.get('rubric_reason', ''),
        )

        # ----- Rating update (replaces the Q-table difficulty ladder) -----
        # Feed the rich rubric score in directly: a 7/10 answer counts as a
        # 0.7 "win" against a question pitched at the current difficulty level.
        # This is the only line that moves the candidate's ability estimate.
        was_clarification = s['last_was_clarification']
        s['micro'].update(topic, level, score / 10.0)

        # The pass / partial / fail buckets now drive ONLY the UI message and
        # the optional clarification follow-up. They no longer hold the math
        # together, so a single low score can't lock anyone out of a topic.
        if score >= 6.0:
            ui_status = ("Clarification accepted. Moving on." if was_clarification
                         else "Strong answer recorded.")
            s['in_clarification_loop'] = False
        elif 4.0 <= score < 6.0 and not s['in_clarification_loop']:
            s['in_clarification_loop'] = True
            ui_status = "Partial answer detected. Stand by for a quick follow-up."
        else:
            s['in_clarification_loop'] = False
            ui_status = "Answer recorded."

        # Topic retirement is now decided by the rating tracker's confidence
        # band (see MacroAgent._is_exhausted) — no manual uncertainty forcing.
        if s['micro'].is_confident(topic):
            print(f"[SYSTEM] Confident in '{topic}' "
                  f"(estimate {s['micro'].get_estimate(topic)}). Macro will pivot away.")

        emit('evaluation_results', {
            'status': ui_status,
            'tech_score': score,
            'confidence_score': conf_score,
            'ideal_answer': ideal,
            'candidate_answer': candidate_text,
            'rubric_reason': text_result.get('rubric_reason', ''),
            'determination': audio_result.get('determination', 0),
            'doubt': audio_result.get('doubt', 0),
            'anxiety': audio_result.get('anxiety', 0),
        })
    except Exception as e:
        print(f"[ERROR] submit_final_answer failed: {e}")
        emit('error', {'message': f'Evaluation failed: {str(e)}'})


@socketio.on('request_next_question')
def handle_next_question(data):
    if not current_user.is_authenticated:
        return
    sid = request.sid
    s = get_session(sid)
    print(f"\n[API] User requested next state (sid={sid})...")

    try:
        if s['in_clarification_loop']:
            print("\n[API] Generating Clarification Follow-up...")
            topic = s['current_topic']
            level = s['current_level']

            q_data = generator.generate_clarification(
                topic, s['last_question'],
                s['last_candidate_answer'], s['ideal_answer']
            )
            s['last_question'] = q_data['question']
            s['ideal_answer'] = q_data['ideal_answer']
            s['last_was_clarification'] = True
            s['macro'].register_question_asked(topic)

            emit('telemetry_update', {
                'status': 'Clarification needed. Please answer the follow-up.',
                'topic': topic,
                'level': level,
                'uncertainty': s['micro'].get_uncertainty(topic),
                'question': s['last_question']
            })
            return

        s['last_was_clarification'] = False
        next_topic = s['macro'].select_next_topic()

        if next_topic:
            next_level = s['micro'].get_current_level(next_topic)
            s['current_topic'] = next_topic
            s['current_level'] = next_level

            q_data = generator.generate_question(next_topic, next_level)
            s['last_question'] = q_data['question']
            s['ideal_answer'] = q_data['ideal_answer']
            s['macro'].register_question_asked(next_topic)

            emit('telemetry_update', {
                'status': 'Awaiting Candidate Audio.',
                'topic': next_topic,
                'level': next_level,
                'uncertainty': s['micro'].get_uncertainty(next_topic),
                'question': s['last_question']
            })
        else:
            # Interview complete
            print("\n[API] Interview Complete. Generating final report and metrics...")

            full_timeline_prompt = s['report_manager'].build_comprehensive_prompt()
            final_report_text = generator.generate_final_report(full_timeline_prompt)
            s['cached_summary'] = final_report_text

            history = s['report_manager'].interview_history
            # Final ability estimates per skill: {topic: (rating, sigma)}.
            estimates = s['micro'].get_all_estimates()

            if history:
                avg_tech_score = sum(h['tech_score'] for h in history) / len(history)
                avg_confidence = sum(h['confidence_score'] for h in history) / len(history)
                avg_doubt = sum(h['doubt'] for h in history) / len(history)
            else:
                avg_tech_score = avg_confidence = avg_doubt = 0

            # Skill depth = average of the final per-skill ability ratings.
            avg_final_level = (sum(rating for rating, _sigma in estimates.values()) / len(estimates)
                               if estimates else 0)

            radar_metrics = [
                round(avg_tech_score * 10),
                round(avg_confidence * 10),
                round(max(0, min(100, (1 - avg_doubt) * 100))),
                round((avg_final_level / 10) * 100),
            ]

            db.complete_interview(
                interview_id=s['interview_id'],
                final_report=final_report_text,
                radar_metrics=radar_metrics
            )

            emit('telemetry_update', {
                'status': 'INTERVIEW COMPLETE. ALL CEILINGS LOCKED.',
                'topic': 'DONE',
                'level': '-',
                'uncertainty': 0,
                'question': 'Thank you for your time. The interview is now concluded.',
                'final_report': final_report_text,
                'radar_data': radar_metrics,
                'radar_labels': [
                    'Technical Accuracy',
                    'Vocal Confidence',
                    'Clarity',
                    'Skill Depth'
                ]
            })
    except Exception as e:
        print(f"[ERROR] request_next_question failed: {e}")
        emit('error', {'message': f'Next-question step failed: {str(e)}'})


if __name__ == '__main__':
    print("Starting AI Interviewer Server on http://127.0.0.1:5000")
    socketio.run(app, debug=True)
