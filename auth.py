"""
Authentication module: signup, login, logout, current-user binding.

Drop this file alongside app.py and db_manager.py. Wire it up in app.py
with three lines:

    from auth import auth_bp, login_manager, init_auth
    login_manager.init_app(app)
    app.register_blueprint(auth_bp)
    init_auth(db_manager_instance)  # share the existing DatabaseManager singleton

Then protect routes with @login_required and use current_user.id wherever
you need to bind interview data to the logged-in user.
"""
import re
import bcrypt
from flask import Blueprint, request, render_template, redirect, url_for, flash
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)

auth_bp = Blueprint('auth', __name__)
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access the interviewer.'
login_manager.login_message_category = 'info'

# Shared DatabaseManager instance — injected by app.py via init_auth().
# Auth shouldn't open its own DB connection; it reuses the one app.py created.
_db = None
def init_auth(db_manager):
    """Called once from app.py to wire the shared DB instance."""
    global _db
    _db = db_manager


# ---------------------------------------------------------------------------
# User class (Flask-Login contract)
# ---------------------------------------------------------------------------
class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

    def get_id(self):
        # Flask-Login expects a string id
        return str(self.id)


@login_manager.user_loader
def load_user(user_id):
    """Rehydrate the User object from the session-stored id on every request."""
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return None
    if _db is None:
        return None
    row = _db.get_user_by_id(uid)
    if not row:
        return None
    return User(id=row['id'], username=row['username'], email=row['email'])


# ---------------------------------------------------------------------------
# Password hashing — bcrypt with a sensible work factor
# ---------------------------------------------------------------------------
def hash_password(plaintext: str) -> str:
    """Hash a plaintext password with bcrypt. Work factor 12 ≈ 250ms per hash."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(plaintext.encode('utf-8'), salt).decode('utf-8')


def verify_password(plaintext: str, stored_hash: str) -> bool:
    """Constant-time bcrypt verification. Returns False on any failure mode."""
    try:
        return bcrypt.checkpw(
            plaintext.encode('utf-8'),
            stored_hash.encode('utf-8')
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
USERNAME_RE = re.compile(r'^[a-zA-Z0-9_]{3,30}$')

def _validate_signup(username, email, password):
    if not username or not USERNAME_RE.match(username):
        return "Username must be 3-30 characters, letters/numbers/underscore only."
    if not email or not EMAIL_RE.match(email):
        return "Please enter a valid email address."
    if not password or len(password) < 8:
        return "Password must be at least 8 characters."
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        err = _validate_signup(username, email, password)
        if err:
            flash(err, 'error')
            return render_template('signup.html', username=username, email=email)

        # Uniqueness checks
        if _db.get_user_by_username(username):
            flash('That username is already taken.', 'error')
            return render_template('signup.html', email=email)
        if _db.get_user_by_email(email):
            flash('That email is already registered.', 'error')
            return render_template('signup.html', username=username)

        password_hash = hash_password(password)
        user_id = _db.create_user(username, email, password_hash)
        if not user_id:
            flash('Could not create the account. Please try again.', 'error')
            return render_template('signup.html', username=username, email=email)

        # Log the new user in immediately
        user = User(id=user_id, username=username, email=email)
        login_user(user, remember=True)
        return redirect(url_for('index'))

    return render_template('signup.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()
        password = request.form.get('password', '')

        if not identifier or not password:
            flash('Enter your username/email and password.', 'error')
            return render_template('login.html', identifier=identifier)

        # Allow either username or email as the identifier
        row = (_db.get_user_by_username(identifier)
               or _db.get_user_by_email(identifier.lower()))
        if not row or not verify_password(password, row['password_hash']):
            # Same error for "no such user" and "wrong password" so attackers
            # can't enumerate valid usernames.
            flash('Invalid username/email or password.', 'error')
            return render_template('login.html', identifier=identifier)

        user = User(id=row['id'], username=row['username'], email=row['email'])
        login_user(user, remember=True)

        next_url = request.args.get('next')
        return redirect(next_url or url_for('index'))

    return render_template('login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))


@auth_bp.route('/my-interviews')
@login_required
def my_interviews():
    interviews = _db.get_user_interviews(current_user.id)

    # Trend data (oldest->newest) for the chart at the top of the page.
    # Vocal confidence is resume-independent; tech is the overall average.
    trend = []
    for r in _db.get_user_trend_data(current_user.id):
        trend.append({
            'date': r['session_date'].strftime('%b %d, %Y') if r.get('session_date') else '',
            'tech': round(float(r['avg_tech'] or 0), 1),
            'conf': round(float(r['avg_conf'] or 0), 1),
        })

    return render_template(
        'my_interviews.html',
        interviews=interviews,
        username=current_user.username,
        trend=trend,
    )
