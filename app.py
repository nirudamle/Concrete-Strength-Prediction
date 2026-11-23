from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import pickle
import numpy as np
import sqlite3
import os
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')


model_path = 'concrete_strength_predict.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)


DATABASE = 'users.db'

def get_db():
    """Open a new database connection."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row   
    return conn

def init_db():
    """Create tables if they don't exist yet."""
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name    TEXT    NOT NULL,
                last_name     TEXT    NOT NULL,
                email         TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                role          TEXT    DEFAULT '',
                created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()


init_db()


def login_required(f):
    """Decorator – redirects to login if user is not in session."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_email' not in session:
            flash('Please log in to access that page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def get_user_by_email(email):
    """Return a user row by email, or None."""
    with get_db() as conn:
        return conn.execute(
            'SELECT * FROM users WHERE email = ?', (email,)
        ).fetchone()

def create_user(first_name, last_name, email, password, role):
    """Insert a new user and return True, or return False if email exists."""
    try:
        with get_db() as conn:
            conn.execute(
                '''INSERT INTO users (first_name, last_name, email, password_hash, role)
                   VALUES (?, ?, ?, ?, ?)''',
                (first_name, last_name, email,
                 generate_password_hash(password), role)
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        # UNIQUE constraint on email failed
        return False

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_email' in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        first_name       = request.form.get('first_name', '').strip()
        last_name        = request.form.get('last_name', '').strip()
        email            = request.form.get('email', '').strip().lower()
        password         = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        role             = request.form.get('role', '')
        terms            = request.form.get('terms')

        
        if not all([first_name, last_name, email, password, confirm_password]):
            flash('All fields are required.', 'error')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')

        if len(password) < 8:
            flash('Password must be at least 8 characters.', 'error')
            return render_template('register.html')

        if not terms:
            flash('You must agree to the Terms of Service.', 'error')
            return render_template('register.html')

        
        success = create_user(first_name, last_name, email, password, role)
        if not success:
            flash('An account with that email already exists.', 'error')
            return render_template('register.html')

        flash(f'Account created! Welcome, {first_name}. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_email' in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        remember = request.form.get('remember')

        user = get_user_by_email(email)

        if user and check_password_hash(user['password_hash'], password):
            session['user_email']      = user['email']
            session['user_first_name'] = user['first_name']
            session.permanent          = bool(remember)
            flash(f"Welcome back, {user['first_name']}!", 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))
        else:
            flash('Invalid email or password.', 'error')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    name = session.get('user_first_name', '')
    session.clear()
    flash(f'You have been logged out, {name}.', 'info')
    return redirect(url_for('home'))



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            cement           = float(request.form['cement'])
            slag             = float(request.form['slag'])
            fly_ash          = float(request.form['fly_ash'])
            water            = float(request.form['water'])
            superplasticizer = float(request.form['superplasticizer'])
            coarse_agg       = float(request.form['coarse_agg'])
            fine_agg         = float(request.form['fine_agg'])
            age              = int(request.form['age'])

            input_data = np.array([[cement, slag, fly_ash, water, superplasticizer,
                                    coarse_agg, fine_agg, age]])

            prediction = model.predict(input_data)[0]

            if prediction < 20:
                category    = "Low Strength"
                badge_class = "badge bg-danger"
            elif prediction < 40:
                category    = "Medium Strength"
                badge_class = "badge bg-warning text-dark"
            elif prediction < 60:
                category    = "High Strength"
                badge_class = "badge bg-success"
            else:
                category    = "Ultra High Strength"
                badge_class = "badge bg-primary"

            return render_template('predict.html',
                                   prediction=round(prediction, 2),
                                   category=category,
                                   badge_class=badge_class,
                                   form_data=request.form)

        except Exception as e:
            return render_template('predict.html',
                                   error=f"Error in prediction: {str(e)}")

    return render_template('predict.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data       = request.get_json()
        input_data = np.array([[
            data['cement'],
            data['slag'],
            data['fly_ash'],
            data['water'],
            data['superplasticizer'],
            data['coarse_agg'],
            data['fine_agg'],
            data['age']
        ]])
        prediction = model.predict(input_data)[0]
        return jsonify({'prediction': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)