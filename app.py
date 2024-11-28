# app.py

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import (
    LoginManager, login_user, login_required,
    logout_user, current_user
)
from flask_bcrypt import Bcrypt
from flask_mail import Mail, Message
from models import db, User
from forms import (
    RegistrationForm, LoginForm,
    RequestResetForm, ResetPasswordForm
)
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'  # Replace with a secure key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

# Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Use your email provider's SMTP server
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'feroarshad692@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'FEROZ18@55%'     # Replace with your email password

db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
mail = Mail(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

# Load the dataset
movies = pd.read_csv('data/movies.csv')

# Optionally, reduce the dataset size
# Uncomment the following line to reduce the dataset size
# movies = movies.sample(5000, random_state=42).reset_index(drop=True)

# Preprocess the data
def preprocess_data(movies_df):
    # Fill NaN values with empty strings
    movies_df['genres'] = movies_df['genres'].fillna('')
    # Combine relevant features
    movies_df['combined_features'] = movies_df['genres']
    return movies_df

movies = preprocess_data(movies)

# Feature extraction using TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

# Reset index of the dataframe and create a reverse mapping of indices and movie titles
movies = movies.reset_index()
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_recommendations(title, tfidf_matrix=tfidf_matrix):
    # Get the index of the movie that matches the title
    idx = indices.get(title)
    if idx is None:
        return pd.DataFrame()
    # Compute the cosine similarity between the target movie and all others
    cosine_similarities = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    # Get the scores of the top 10 most similar movies
    sim_scores = list(enumerate(cosine_similarities))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the indices of the top 10 most similar movies (excluding itself)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return movies.iloc[movie_indices]

# Extract unique genres
def get_unique_genres(movies_df):
    genres_set = set()
    for genres in movies_df['genres']:
        for genre in genres.split('|'):
            genres_set.add(genre.strip())
    return sorted(list(genres_set))

genres_list = get_unique_genres(movies)

@app.context_processor
def inject_genres():
    return dict(genres=genres_list)

@app.route('/')
def index():
    # Random movies
    random_movies = movies.sample(12).to_dict(orient='records')
    return render_template(
        'index.html',
        random_movies=random_movies
    )

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_pw = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(
            username=form.username.data,
            email=form.email.data,
            password=hashed_pw
        )
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            return redirect(url_for('index'))
        else:
            flash('Login Unsuccessful. Please check email and password.', 'danger')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '').strip()
    genre = request.args.get('genre', '')
    
    results = movies
    
    if query:
        # If query is provided, filter by title
        results = results[results['title'].str.contains(query, case=False, na=False)]
    
    if genre:
        # If genre is selected, filter by genre
        results = results[results['genres'].str.contains(genre, case=False, na=False)]
    
    results = results.to_dict(orient='records')
    
    return render_template(
        'search_results.html',
        query=query,
        genre=genre,
        movies=results
    )

@app.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    # Get the movie details
    movie = movies.loc[movies['index'] == movie_id]
    if movie.empty:
        flash('Movie not found.', 'warning')
        return redirect(url_for('index'))
    movie = movie.iloc[0]
    # Get recommendations
    recommendations = get_recommendations(movie['title'])
    recommendations = recommendations.to_dict(orient='records')
    return render_template('movie_detail.html', movie=movie, recommendations=recommendations)

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            send_reset_email(user)
            flash(
                'An email has been sent with instructions to reset your password.',
                'info'
            )
            return redirect(url_for('login'))
        else:
            flash('No account with that email exists.', 'warning')
    return render_template('reset_request.html', form=form)

def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message(
        'Password Reset Request',
        sender='noreply@mrsaai.com',
        recipients=[user.email]
    )
    msg.body = f'''To reset your password, visit the following link:
{url_for('reset_token', token=token, _external=True)}

If you did not make this request, simply ignore this email.
'''
    mail.send(msg)

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token.', 'warning')
        return redirect(url_for('reset_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_pw = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user.password = hashed_pw
        db.session.commit()
        flash('Your password has been updated!', 'success')
        return redirect(url_for('login'))
    return render_template('reset_token.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
