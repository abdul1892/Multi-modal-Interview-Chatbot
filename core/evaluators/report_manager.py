import os
import re
from collections import defaultdict
from urllib.parse import quote_plus

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.widgets.markers import makeMarker


# ---------------------------------------------------------------------------
# Common interview lead-in patterns that don't carry concept signal — stripped
# from the question text before constructing the GFG search URL.
# ---------------------------------------------------------------------------
_LEAD_IN_PATTERNS = [
    r'^(can|could|would)\s+you\s+(please\s+|kindly\s+)?',
    r'^please\s+',
    r'^kindly\s+',
    r'^(briefly\s+|in\s+detail\s+)?'
    r'(describe|explain|tell\s+me\s+about|tell\s+me|walk\s+me\s+through|'
    r'what\s+(is|are|do|does)|how\s+(does|do|would|is)|why\s+(do|does|is))\s+',
]


def _gfg_link_for_question(question, topic):
    """
    Build a GeeksforGeeks search URL targeting the specific concept of the
    question, not just the broad topic. Returns (display_label, url).

    Search URLs always resolve to something, so there's zero hallucination
    risk — even for an exotic concept GFG hasn't covered, the user lands on
    a search results page rather than a 404.
    """
    if not question:
        question = topic or "interview question"

    cleaned = question.strip()
    for pat in _LEAD_IN_PATTERNS:
        cleaned = re.sub(pat, '', cleaned, count=1, flags=re.IGNORECASE)
    # Strip punctuation, collapse whitespace
    cleaned = re.sub(r'[?!.,;:"\']', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Cap query length to keep URLs sane
    if len(cleaned) > 90:
        cleaned = cleaned[:90].rsplit(' ', 1)[0]

    # Make sure the topic is part of the query — that's how the user disambiguates
    # "memory management in Go" from "memory management in C"
    if topic and topic.lower() not in cleaned.lower():
        query = f"{topic} {cleaned}"
    else:
        query = cleaned

    url = f"https://www.geeksforgeeks.org/search/?gq={quote_plus(query)}"
    display = cleaned if len(cleaned) <= 70 else cleaned[:67] + '…'
    return display, url


def _escape_for_paragraph(text):
    """Escape XML-special chars so ReportLab Paragraph doesn't choke on '<', '>', '&'."""
    if not text:
        return ""
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;'))


def _sanitize_for_reportlab(text):
    """
    Convert LLM-emitted markdown into ReportLab-friendly inline markup, and
    strip stray asterisks that the model insists on producing despite being
    told not to. Belt-and-suspenders so the PDF is clean even if the prompt
    is partially ignored.
    """
    if not text:
        return ""
    # **bold** → <b>bold</b>
    text = re.sub(r'\*\*([^*\n]+?)\*\*', r'<b>\1</b>', text)
    # Markdown links [label](url) → ReportLab <link>
    text = re.sub(
        r'\[([^\]]+)\]\(([^)]+)\)',
        r'<link href="\2"><font color="#2563eb">\1</font></link>',
        text
    )
    # Drop leading "* " bullet markers
    text = re.sub(r'(?m)^\s*\*\s+', '', text)
    # Drop any remaining stray asterisks
    text = text.replace('*', '')
    # Escape ampersands so reportlab doesn't choke on stray "&"
    text = text.replace('&', '&amp;')
    # Re-allow the tags we just inserted
    text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
    return text


def _parse_sections(raw):
    """
    Split the LLM's response into (title, body) pairs using our [SECTION N: TITLE]
    markers. If the model ignored the markers entirely, return the whole thing
    as a single untitled section so we still produce a valid PDF.
    """
    if not raw:
        return []
    pattern = re.compile(r'\[SECTION\s+\d+:\s*([^\]]+)\]', re.IGNORECASE)
    matches = list(pattern.finditer(raw))
    if not matches:
        return [(None, raw.strip())]
    sections = []
    for i, m in enumerate(matches):
        title = m.group(1).strip().title()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[body_start:body_end].strip()
        sections.append((title, body))
    return sections


def _delta_markup(delta, suffix=''):
    """Colored up/down/flat change string. Higher is always better for our metrics."""
    if delta > 0.05:
        color, arrow = '#15803d', '&#9650;'      # green up
    elif delta < -0.05:
        color, arrow = '#b91c1c', '&#9660;'      # red down
    else:
        color, arrow = '#64748b', '&#9654;'      # gray flat
    sign = '+' if delta > 0 else ''
    return f"<font color='{color}'>{arrow} {sign}{delta}{suffix}</font>"


def _trend_line_drawing(trend):
    """
    A 0-10 line chart of Technical (blue) and Vocal Confidence (green) across
    every completed interview. Built with ReportLab's own graphics so we add no
    new dependency.
    """
    width, height = 480, 210
    d = Drawing(width, height)

    lc = HorizontalLineChart()
    lc.x, lc.y = 45, 38
    lc.width, lc.height = width - 80, height - 80
    lc.data = [[t['tech'] for t in trend], [t['conf'] for t in trend]]

    # Thin x-axis labels if there are a lot of sessions, so they don't collide.
    names = [t['date'] for t in trend]
    if len(names) > 8:
        step = (len(names) // 8) + 1
        names = [nm if (i % step == 0 or i == len(names) - 1) else ''
                 for i, nm in enumerate(names)]
    lc.categoryAxis.categoryNames = names
    lc.categoryAxis.labels.fontSize = 7
    lc.categoryAxis.labels.fillColor = colors.HexColor('#475569')

    lc.valueAxis.valueMin = 0
    lc.valueAxis.valueMax = 10
    lc.valueAxis.valueStep = 2
    lc.valueAxis.labels.fontSize = 7
    lc.valueAxis.labels.fillColor = colors.HexColor('#475569')

    lc.lines[0].strokeColor = colors.HexColor('#2563eb')   # technical
    lc.lines[1].strokeColor = colors.HexColor('#10b981')   # confidence
    lc.lines[0].strokeWidth = 2
    lc.lines[1].strokeWidth = 2
    lc.lines[0].symbol = makeMarker('FilledCircle')
    lc.lines[1].symbol = makeMarker('FilledCircle')
    lc.lines[0].symbol.size = 4
    lc.lines[1].symbol.size = 4
    d.add(lc)

    d.add(String(45, height - 16, 'Performance Trend (0\u201310 scale)',
                 fontSize=11, fillColor=colors.HexColor('#0f172a')))
    d.add(String(45, 14, '\u25CF Technical',
                 fontSize=8, fillColor=colors.HexColor('#2563eb')))
    d.add(String(140, 14, '\u25CF Vocal Confidence',
                 fontSize=8, fillColor=colors.HexColor('#10b981')))
    return d


def _build_progress_section(progress, h2_style, h3_style, body_style):
    """
    Build the 'Progress & Trends' flowables. All numbers are pre-computed in
    app.build_progress_context(); this function only renders them.
    """
    flow = [Paragraph("Progress &amp; Trends", h2_style)]

    if not progress or progress.get('is_first'):
        flow.append(Paragraph(
            "This is the first completed interview on this account. These "
            "results establish the baseline that future sessions will be "
            "measured against — your next report will show how you've changed "
            "since today.",
            body_style
        ))
        return flow

    cmp = progress['comparison']

    # ----- Headline comparison vs the previous interview -----
    flow.append(Paragraph(
        f"Since your last interview ({cmp['prev_date']} &#8594; {cmp['curr_date']})",
        h3_style
    ))

    cell = ParagraphStyle('PCell', parent=body_style, fontSize=10, spaceAfter=0)
    hdr = ParagraphStyle('PHdr', parent=cell, textColor=colors.whitesmoke,
                         fontName='Helvetica-Bold')

    rows = [[
        Paragraph("Metric", hdr), Paragraph("Previous", hdr),
        Paragraph("This session", hdr), Paragraph("Change", hdr),
    ]]
    rows.append([
        Paragraph("Technical score (overall)", cell),
        Paragraph(f"{cmp['tech_prev']}/10", cell),
        Paragraph(f"{cmp['tech_curr']}/10", cell),
        Paragraph(_delta_markup(cmp['tech_delta']), cell),
    ])
    rows.append([
        Paragraph("Vocal confidence", cell),
        Paragraph(f"{cmp['conf_prev']}/10", cell),
        Paragraph(f"{cmp['conf_curr']}/10", cell),
        Paragraph(_delta_markup(cmp['conf_delta']), cell),
    ])
    tbl = Table(rows, colWidths=[210, 90, 100, 90])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    flow.append(tbl)
    flow.append(Spacer(1, 8))

    # ----- Per-skill, like-for-like (only skills present in BOTH sessions) -----
    flow.append(Paragraph("Skills carried over from last time", h3_style))
    if cmp['carried']:
        for c in cmp['carried']:
            flow.append(Paragraph(
                f"<b>{c['topic']}</b>: {c['prev']}/10 &#8594; {c['curr']}/10 &nbsp; "
                + _delta_markup(c['delta']),
                body_style
            ))
    else:
        flow.append(Paragraph(
            "No skills overlapped between the two sessions, so there's no "
            "like-for-like technical comparison this time. The resume-independent "
            "metrics above still reflect real change.",
            body_style
        ))

    if cmp['new_skills']:
        flow.append(Paragraph(
            "<font color='#475569'><b>New this session (baseline set):</b> "
            + ", ".join(cmp['new_skills']) + "</font>",
            body_style
        ))
    if cmp['dropped_skills']:
        flow.append(Paragraph(
            "<font color='#94a3b8'>Not revisited this session: "
            + ", ".join(cmp['dropped_skills']) + "</font>",
            body_style
        ))

    # ----- Full trend across all interviews -----
    trend = progress.get('trend') or []
    if len(trend) >= 2:
        flow.append(Spacer(1, 10))
        flow.append(Paragraph("Across all interviews", h3_style))
        flow.append(_trend_line_drawing(trend))

    return flow


class InterviewReportManager:
    def __init__(self):
        self.interview_history = []
        self.candidate_username = "Candidate"

    def reset(self):
        self.interview_history = []
        self.candidate_username = "Candidate"

    def set_candidate(self, username):
        """Called once per interview to bind the report to the logged-in user."""
        if username:
            self.candidate_username = username

    def log_round(self, question, transcript, tech_score, audio_analytics,
                  topic=None, ideal_answer=None, rubric_reason=None):
        self.interview_history.append({
            'round': len(self.interview_history) + 1,
            'topic': topic or 'General',
            'question': question,
            'ideal_answer': ideal_answer or '',
            'transcript': transcript if transcript.strip() else "[No Response Recorded]",
            'tech_score': tech_score,
            'rubric_reason': rubric_reason or '',
            'determination': audio_analytics.get('determination', 0.0),
            'doubt': audio_analytics.get('doubt', 0.0),
            'anxiety': audio_analytics.get('anxiety', 0.0),
            'confidence_score': audio_analytics.get('confidence_score', 5.0)
        })

    # ---------- Aggregations used by the report layer ----------
    def _topic_average_scores(self):
        by_topic = defaultdict(list)
        for r in self.interview_history:
            by_topic[r['topic']].append(r['tech_score'])
        return {t: sum(scores) / len(scores) for t, scores in by_topic.items()}

    def _weak_topics(self, threshold=3.0):
        """Topics whose average tech score falls strictly below the threshold."""
        return [t for t, s in self._topic_average_scores().items() if s < threshold]

    # ---------- Prompt construction ----------
    def build_comprehensive_prompt(self):
        timeline_text = ""
        for item in self.interview_history:
            timeline_text += (
                f"\nROUND {item['round']} — Topic: {item['topic']}\n"
                f"Question: {item['question']}\n"
                f"Ideal Answer: {item['ideal_answer']}\n"
                f"Candidate's Answer: {item['transcript']}\n"
                f"Technical Score: {item['tech_score']}/10\n"
            )

        username = self.candidate_username
        prompt = f"""You are a Lead Technical HR Architect. Analyze this technical interview timeline
and generate a detailed Executive Performance Report for the candidate named {username}.

INTERVIEW TIMELINE:
{timeline_text}

FORMATTING RULES — these are STRICT:
- Use plain prose only. No markdown, no asterisks, no bullet points, no bold markers, no headers other than the section tags below.
- Refer to the candidate as "{username}" throughout. Never write "[Not Provided]" or "the candidate" generically.
- Do NOT mention duration, interview length, or timestamps anywhere.
- Do NOT discuss vocal patterns, prosody, anxiety, doubt, determination, communication skills, soft skills, body language, mentorship, or behavioral coaching. This report is technical-only.
- Do NOT recommend "communication training", "behavioral coaching", "mentorship", or any non-technical intervention.

Produce exactly four sections, each opened with the marker shown in brackets on its own line. Do not invent additional sections.

[SECTION 1: OVERALL ASSESSMENT]
Two to three sentences summarizing {username}'s overall technical aptitude based on their average score across the rounds. State the average score numerically and whether it indicates competence or a gap.

[SECTION 2: TECHNICAL PERFORMANCE BY TOPIC]
For every distinct topic in the timeline, write one short paragraph (two or three sentences) describing exactly what {username} demonstrated and exactly what they failed to demonstrate. Reference specific concepts from the questions and the ideal answers. Do not use bullets — write in flowing paragraphs separated by blank lines.

[SECTION 3: STRENGTHS]
Identify any rounds where {username} scored 6.0 or higher. Briefly describe the specific concept they got right. If every round scored below 6.0, write exactly: "No technical strengths were demonstrated in this session." — and nothing more for this section.

[SECTION 4: HIRING RECOMMENDATION]
Open with one of these exact phrases: "Recommend", "Recommend with reservations", or "Do not recommend at this time." Then justify the decision in two or three sentences referencing only the technical scores and gaps observed. Do not suggest any training program — that is handled separately by the system.
"""
        return prompt

    # ---------- PDF generation ----------
    def generate_pdf(self, filename, executive_summary, progress=None):
        doc = SimpleDocTemplate(
            filename, pagesize=letter,
            rightMargin=50, leftMargin=50,
            topMargin=50, bottomMargin=50
        )
        story = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'DocTitle', parent=styles['Heading1'],
            fontSize=22, leading=26, spaceAfter=4,
            textColor=colors.HexColor('#0f172a')
        )
        subtitle_style = ParagraphStyle(
            'DocSubtitle', parent=styles['BodyText'],
            fontSize=11, leading=14, textColor=colors.HexColor('#64748b'),
            spaceAfter=18
        )
        h2_style = ParagraphStyle(
            'SectionH2', parent=styles['Heading2'],
            fontSize=14, leading=18, spaceBefore=18, spaceAfter=10,
            textColor=colors.HexColor('#2563eb'), keepWithNext=True
        )
        h3_style = ParagraphStyle(
            'SectionH3', parent=styles['Heading3'],
            fontSize=12, leading=16, spaceBefore=10, spaceAfter=6,
            textColor=colors.HexColor('#0f172a'), keepWithNext=True
        )
        body_style = ParagraphStyle(
            'Body', parent=styles['BodyText'],
            fontSize=10.5, leading=15,
            textColor=colors.HexColor('#1e293b'),
            spaceAfter=8
        )
        link_style = ParagraphStyle(
            'LinkRow', parent=body_style,
            leftIndent=14, spaceAfter=3, fontSize=10
        )

        # ===== Title =====
        story.append(Paragraph("AI Interviewer", title_style))
        story.append(Paragraph("Evaluation Finalized &amp; Archived", subtitle_style))

        # ===== Candidate metadata (no Duration row) =====
        meta_data = [
            ["Candidate", self.candidate_username],
            ["Interview Type", "Technical"],
            ["Rounds Completed", str(len(self.interview_history))],
        ]
        meta_table = Table(meta_data, colWidths=[120, 360])
        meta_table.setStyle(TableStyle([
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#64748b')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#0f172a')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10.5),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 8))

        # ===== Progress & Trends (cross-interview comparison) =====
        # Rendered from numbers pre-computed in app.build_progress_context().
        # Omitted entirely if the caller didn't supply progress data.
        if progress is not None:
            for f in _build_progress_section(progress, h2_style, h3_style, body_style):
                story.append(f)
            story.append(Spacer(1, 12))

        # ===== Session Timeline (Round | Topic | Tech | Confidence) =====
        story.append(Paragraph("Session Timeline", h2_style))
        table_data = [["Round", "Topic", "Technical Score", "Confidence Score"]]
        for idx, item in enumerate(self.interview_history):
            table_data.append([
                f"Q{idx + 1}",
                item['topic'],
                f"{item['tech_score']}/10",
                f"{item['confidence_score']}/10",
            ])
        t = Table(table_data, colWidths=[60, 200, 110, 110])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            ('LEFTPADDING', (1, 1), (1, -1), 12),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('FONTSIZE', (0, 1), (-1, -1), 9.5),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

        # ===== LLM-generated narrative =====
        story.append(Paragraph("Executive Evaluation", h2_style))

        sections = _parse_sections(executive_summary or "")
        if not sections:
            story.append(Paragraph("No executive summary was generated.", body_style))
        else:
            for section_title, section_body in sections:
                if section_title:
                    story.append(Paragraph(section_title, h3_style))
                cleaned = _sanitize_for_reportlab(section_body)
                # Split on blank lines to form paragraphs
                for para in re.split(r'\n\s*\n', cleaned):
                    para = para.strip()
                    if not para:
                        continue
                    # Collapse single newlines inside a paragraph into spaces
                    para = re.sub(r'\s*\n\s*', ' ', para)
                    story.append(Paragraph(para, body_style))

        # ===== Per-Question Breakdown =====
        # Mirrors what the candidate saw live in the XAI panel — question,
        # ideal reference, transcript, grader's reasoning, score — plus a
        # GeeksforGeeks search link targeted at the specific concept of the
        # question (not the broad topic).
        if self.interview_history:
            story.append(Spacer(1, 14))
            story.append(Paragraph("Recommended Learning Resources", h2_style))
            story.append(Paragraph(
                "For each question in this interview, the entries below show "
                "what was asked, the AI's ideal reference answer, the "
                "candidate's actual response, the grader's reasoning, the "
                "technical score, and a GeeksforGeeks learning link targeted "
                "at the specific concept covered.",
                body_style
            ))

            label_style = ParagraphStyle(
                'QFieldLabel', parent=body_style,
                fontName='Helvetica-Bold', textColor=colors.HexColor('#475569'),
                fontSize=9.5, spaceAfter=2, spaceBefore=4
            )
            value_style = ParagraphStyle(
                'QFieldValue', parent=body_style,
                fontSize=10, leading=14, spaceAfter=4,
                leftIndent=0, textColor=colors.HexColor('#0f172a')
            )
            grader_style = ParagraphStyle(
                'QGrader', parent=value_style,
                textColor=colors.HexColor('#a16207'), fontName='Helvetica-Oblique'
            )
            question_head_style = ParagraphStyle(
                'QHead', parent=styles['Heading3'],
                fontSize=12, leading=15, spaceBefore=12, spaceAfter=6,
                textColor=colors.HexColor('#0f172a'),
                borderPadding=4, keepWithNext=True
            )
            gfg_style = ParagraphStyle(
                'GFGLink', parent=value_style,
                spaceAfter=8, leftIndent=0
            )

            for item in self.interview_history:
                gfg_label, gfg_url = _gfg_link_for_question(item['question'], item['topic'])

                # Score color: green if strong, amber if moderate, red if weak
                score_val = item['tech_score']
                if score_val >= 6:
                    score_color = '#15803d'
                elif score_val >= 3:
                    score_color = '#a16207'
                else:
                    score_color = '#b91c1c'

                head_html = (
                    f"Round {item['round']}  ·  {_escape_for_paragraph(item['topic'])}  ·  "
                    f"<font color='{score_color}'>Score: {score_val}/10</font>"
                )

                block = [
                    Paragraph(head_html, question_head_style),

                    Paragraph("Question", label_style),
                    Paragraph(_escape_for_paragraph(item['question']), value_style),

                    Paragraph("Ideal Reference", label_style),
                    Paragraph(
                        _escape_for_paragraph(item['ideal_answer'])
                        or "<i>(not recorded)</i>",
                        value_style
                    ),

                    Paragraph("Candidate's Answer", label_style),
                    Paragraph(_escape_for_paragraph(item['transcript']), value_style),
                ]

                if item.get('rubric_reason'):
                    block.append(Paragraph("Grader's Reasoning", label_style))
                    block.append(Paragraph(
                        _escape_for_paragraph(item['rubric_reason']),
                        grader_style
                    ))

                block.append(Paragraph(
                    f"<b>GeeksforGeeks:</b> "
                    f"<link href='{gfg_url}'>"
                    f"<font color='#2563eb'>{_escape_for_paragraph(gfg_label)}</font>"
                    f"</link><br/>"
                    f"<font color='#94a3b8' size='8'>{_escape_for_paragraph(gfg_url)}</font>",
                    gfg_style
                ))

                # KeepTogether tries to keep heading + first few fields on one page.
                # For very long answers, ReportLab will spill the trailing fields
                # onto the next page — that's fine, the heading still anchors the block.
                story.append(KeepTogether(block[:4]))
                story.extend(block[4:])

        doc.build(story)
        print(f"[SYSTEM] Detailed PDF Assessment generated successfully: {filename}")