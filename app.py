import os
import io
import json
import csv
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, Response, abort
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, validators
import cv2
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from PIL import Image
from forms import LoginForm

# Custom metric function
def mae_in_months(y_true, y_pred):
    return tf.keras.backend.abs(y_true - y_pred) * 12

# Register the custom metric
tf.keras.utils.get_custom_objects()['mae_in_months'] = mae_in_months

# Create the model
def create_model():
    # Create base VGG16 model
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)  # VGG16 input size
    )
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1)(x)  # Single output for age prediction
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Print model summary for debugging
    print("\nModel architecture:")
    model.summary()
    
    # Print layer names
    print("\nModel layer names:")
    for layer in model.layers:
        print(f"- {layer.name}")
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mean_absolute_error',
        metrics=[mae_in_months]
    )
    
    return model

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')  # Make sure to set this in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['WTF_CSRF_TIME_LIMIT'] = 3600  # Set CSRF token expiry to 1 hour

# Initialize CSRF protection
csrf = CSRFProtect(app)

CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:8000", "http://127.0.0.1:8000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-CSRFToken"],
        "supports_credentials": True
    }
})

# Add security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Create and load the model
try:
    # Create the model architecture
    model = create_model()
    
    # Print model summary before loading weights
    print("\nModel architecture before loading weights:")
    model.summary()
    
    # Load the pre-trained weights
    model.load_weights('bone_age_weights.best.hdf5')
    print("\nModel weights loaded successfully!")
    
    # Print model summary after loading weights
    print("\nModel architecture after loading weights:")
    model.summary()
    
    # Print layer names for debugging
    print("\nLayer names:")
    for layer in model.layers:
        print(f"- {layer.name}")
except Exception as e:
    print(f"\nError loading model: {str(e)}")
    raise

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    predicted_age = db.Column(db.Float, nullable=False)
    image_filename = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)
    confidence = db.Column(db.Float)  # Optional confidence score
    notes = db.Column(db.Text)  # Optional notes field

    def __init__(self, **kwargs):
        # Convert UTC to IST (UTC+5:30) before storing
        if 'created_at' not in kwargs:
            ist_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
            kwargs['created_at'] = ist_time
        super(Prediction, self).__init__(**kwargs)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def preprocess_image(image_data):
    # Convert bytes to PIL Image
    img = Image.open(io.BytesIO(image_data))
    
    # Convert grayscale to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to VGG16 input size (224x224)
    img = img.resize((224, 224))
    
    # Convert to numpy array and preprocess
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # VGG16-specific preprocessing
    
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

class SignupForm(FlaskForm):
    username = StringField('Username', [validators.DataRequired(), validators.Length(min=4, max=25)])
    email = StringField('Email', [validators.DataRequired(), validators.Email()])
    password = PasswordField('Password', [validators.DataRequired(), validators.Length(min=6)])
    submit = SubmitField('Sign Up')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = form.password.data
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('signup'))
        
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('signup.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password_hash, form.password.data):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'error')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

# Reference ranges for bone age by gender (in years)
BONE_AGE_RANGES = {
    'male': {
        'p3': [0.67, 1.32, 2.14, 3.22, 4.27, 5.31, 6.37, 7.45, 8.53, 9.61, 10.69, 11.77, 12.85, 13.93, 15.01, 16.09, 17.17],
        'p10': [0.84, 1.54, 2.41, 3.52, 4.59, 5.64, 6.71, 7.79, 8.87, 9.95, 11.03, 12.11, 13.19, 14.27, 15.35, 16.43, 17.51],
        'p25': [1.07, 1.84, 2.77, 3.91, 4.99, 6.05, 7.13, 8.21, 9.29, 10.37, 11.45, 12.53, 13.61, 14.69, 15.77, 16.85, 17.93],
        'p50': [1.42, 2.27, 3.27, 4.44, 5.53, 6.60, 7.68, 8.76, 9.84, 10.92, 12.00, 13.08, 14.16, 15.24, 16.32, 17.40, 18.48],
        'p75': [1.77, 2.70, 3.77, 4.97, 6.07, 7.15, 8.23, 9.31, 10.39, 11.47, 12.55, 13.63, 14.71, 15.79, 16.87, 17.95, 19.03],
        'p90': [2.00, 3.00, 4.13, 5.36, 6.47, 7.56, 8.65, 9.73, 10.81, 11.89, 12.97, 14.05, 15.13, 16.21, 17.29, 18.37, 19.45],
        'p97': [2.17, 3.22, 4.40, 5.66, 6.79, 7.89, 8.99, 10.07, 11.15, 12.23, 13.31, 14.39, 15.47, 16.55, 17.63, 18.71, 19.79]
    },
    'female': {
        'p3': [0.67, 1.32, 2.14, 3.22, 4.27, 5.31, 6.37, 7.45, 8.53, 9.61, 10.69, 11.77, 12.85, 13.93, 15.01, 16.09],
        'p10': [0.84, 1.54, 2.41, 3.52, 4.59, 5.64, 6.71, 7.79, 8.87, 9.95, 11.03, 12.11, 13.19, 14.27, 15.35, 16.43],
        'p25': [1.07, 1.84, 2.77, 3.91, 4.99, 6.05, 7.13, 8.21, 9.29, 10.37, 11.45, 12.53, 13.61, 14.69, 15.77, 16.85],
        'p50': [1.42, 2.27, 3.27, 4.44, 5.53, 6.60, 7.68, 8.76, 9.84, 10.92, 12.00, 13.08, 14.16, 15.24, 16.32, 17.40],
        'p75': [1.77, 2.70, 3.77, 4.97, 6.07, 7.15, 8.23, 9.31, 10.39, 11.47, 12.55, 13.63, 14.71, 15.79, 16.87, 17.95],
        'p90': [2.00, 3.00, 4.13, 5.36, 6.47, 7.56, 8.65, 9.73, 10.81, 11.89, 12.97, 14.05, 15.13, 16.21, 17.29, 18.37],
        'p97': [2.17, 3.22, 4.40, 5.66, 6.79, 7.89, 8.99, 10.07, 11.15, 12.23, 13.31, 14.39, 15.47, 16.55, 17.63, 18.71]
    }
}

def generate_gradcam(model, image, layer_name='block5_conv3'):
    """Generate Grad-CAM visualization for the given image"""
    # Get the last convolutional layer
    last_conv_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
    
    if last_conv_layer is None:
        raise ValueError("No convolutional layer found in the model")
    
    grad_model = Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, 0]
    
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
    
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
    
    cam = cv2.resize(cam.numpy(), (224, 224))
    cam = np.maximum(cam, 0) / np.max(cam)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    
    return heatmap

def calculate_confidence(prediction, gender):
    """Calculate confidence score based on prediction and reference ranges"""
    age = float(prediction)
    ranges = BONE_AGE_RANGES[gender.lower()]
    
    # Find the closest age range
    closest_idx = min(range(len(ranges['p50'])), 
                     key=lambda i: abs(ranges['p50'][i] - age))
    
    # Calculate confidence based on percentile ranges
    p3_p97_range = ranges['p97'][closest_idx] - ranges['p3'][closest_idx]
    distance_from_median = abs(age - ranges['p50'][closest_idx])
    
    # Convert to confidence score (0-1)
    confidence = max(0, min(1, 1 - (distance_from_median / (p3_p97_range / 2))))
    
    return confidence

def calculate_percentile(age, gender):
    """Calculate age percentile based on reference ranges"""
    ranges = BONE_AGE_RANGES[gender.lower()]
    
    # Find the closest age range
    closest_idx = min(range(len(ranges['p50'])), 
                     key=lambda i: abs(ranges['p50'][i] - age))
    
    percentiles = [3, 10, 25, 50, 75, 90, 97]
    values = [ranges[f'p{p}'][closest_idx] for p in percentiles]
    
    # Find where the age falls in the percentile range
    for i, value in enumerate(values):
        if age <= value:
            if i == 0:
                return percentiles[0]
            elif i == len(values) - 1:
                return percentiles[-1]
            else:
                # Interpolate between percentiles
                p1, p2 = percentiles[i-1], percentiles[i]
                v1, v2 = values[i-1], values[i]
                return p1 + (p2 - p1) * (age - v1) / (v2 - v1)
    
    return percentiles[-1]

def generate_pdf_report(prediction_id):
    """Generate a PDF report for a prediction"""
    prediction = Prediction.query.get_or_404(prediction_id)
    
    # Create PDF buffer
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph("Bone Age Prediction Report", title_style))
    story.append(Spacer(1, 12))
    
    # Patient Information
    story.append(Paragraph(f"Date: {prediction.created_at.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"Patient ID: {prediction.user_id}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Prediction Results
    story.append(Paragraph("Prediction Results", styles['Heading2']))
    story.append(Paragraph(f"Predicted Age: {prediction.predicted_age:.1f} years", styles['Normal']))
    if prediction.confidence:
        story.append(Paragraph(f"Confidence: {prediction.confidence:.1%}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Add X-ray image
    img_path = os.path.join('static', 'predictions', prediction.image_filename)
    if os.path.exists(img_path):
        img = RLImage(img_path, width=400, height=400)
        story.append(img)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'image' not in request.files:
        return {'error': 'No image uploaded'}, 400
    
    image_file = request.files['image']
    gender = request.form.get('gender', 'male').lower()  # Default to male if not specified
    
    if not image_file.filename:
        return {'error': 'No file selected'}, 400
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if '.' not in image_file.filename or \
       image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return {'error': 'Invalid file type'}, 400
    
    try:
        # Save the image
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"prediction_{timestamp}_{image_file.filename}"
        image_path = os.path.join('static', 'predictions', filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image_file.save(image_path)
        
        # Read image for prediction
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Generate Grad-CAM visualization
        heatmap = generate_gradcam(model, processed_image)
        heatmap_path = os.path.join('static', 'predictions', f'heatmap_{timestamp}.png')
        cv2.imwrite(heatmap_path, heatmap)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        predicted_age = float(prediction[0])
        
        # Calculate confidence and percentile
        confidence = calculate_confidence(predicted_age, gender)
        percentile = calculate_percentile(predicted_age, gender)
        
        # Store prediction in database
        new_prediction = Prediction(
            user_id=current_user.id,
            predicted_age=predicted_age,
            image_filename=filename,
            confidence=confidence,
            notes=json.dumps({
                'gender': gender,
                'percentile': percentile,
                'heatmap_filename': f'heatmap_{timestamp}.png'
            })
        )
        db.session.add(new_prediction)
        db.session.commit()
        
        return {
            'predicted_age': round(predicted_age, 1),
            'confidence': round(confidence * 100, 1),
            'percentile': round(percentile, 1),
            'success': True,
            'prediction_id': new_prediction.id,
            'heatmap_url': url_for('static', filename=f'predictions/heatmap_{timestamp}.png')
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {'error': str(e)}, 500

@app.route('/predictions')
@login_required
def predictions():
    user_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    return render_template('predictions.html', predictions=user_predictions)

@app.route('/export_predictions')
@login_required
def export_predictions():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Date', 'Predicted Age', 'Confidence', 'Gender', 'Percentile', 'Notes'])
    
    for pred in predictions:
        notes = json.loads(pred.notes) if pred.notes else {}
        writer.writerow([
            pred.id,
            pred.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            f"{pred.predicted_age:.1f}",
            f"{pred.confidence * 100:.1f}%",
            notes.get('gender', 'N/A'),
            f"{notes.get('percentile', 0):.0f}",
            notes.get('text', '')
        ])
    
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment;filename=predictions_{datetime.now().strftime("%Y%m%d")}.csv'}
    )

@app.route('/download_report/<int:prediction_id>')
@login_required
def download_report(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    if prediction.user_id != current_user.id:
        abort(403)
    
    # Create PDF report
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    elements.append(Paragraph(f"Bone Age Prediction Report", styles['Title']))
    elements.append(Spacer(1, 12))
    
    # Patient Information
    elements.append(Paragraph("Patient Information", styles['Heading1']))
    elements.append(Paragraph(f"Prediction ID: {prediction.id}", styles['Normal']))
    elements.append(Paragraph(f"Date: {prediction.created_at.strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    
    # Prediction Results
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Prediction Results", styles['Heading1']))
    elements.append(Paragraph(f"Predicted Age: {prediction.predicted_age:.1f} years", styles['Normal']))
    elements.append(Paragraph(f"Confidence: {prediction.confidence * 100:.1f}%", styles['Normal']))
    
    notes = json.loads(prediction.notes) if prediction.notes else {}
    if notes:
        elements.append(Paragraph(f"Gender: {notes.get('gender', 'N/A')}", styles['Normal']))
        elements.append(Paragraph(f"Percentile: {notes.get('percentile', 0):.0f}", styles['Normal']))
        if 'text' in notes:
            elements.append(Paragraph(f"Notes: {notes['text']}", styles['Normal']))
    
    # Add images
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Images", styles['Heading1']))
    
    # Original X-ray
    img_path = os.path.join(app.static_folder, 'predictions', prediction.image_filename)
    if os.path.exists(img_path):
        img = RLImage(img_path, width=250, height=250)
        elements.append(img)
        elements.append(Paragraph("Original X-ray", styles['Normal']))
    
    # Heatmap
    if notes.get('heatmap_filename'):
        heatmap_path = os.path.join(app.static_folder, 'predictions', notes['heatmap_filename'])
        if os.path.exists(heatmap_path):
            elements.append(Spacer(1, 12))
            img = RLImage(heatmap_path, width=250, height=250)
            elements.append(img)
            elements.append(Paragraph("Model Focus Heatmap", styles['Normal']))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    
    return Response(
        buffer.getvalue(),
        mimetype='application/pdf',
        headers={'Content-Disposition': f'attachment;filename=bone_age_report_{prediction.id}.pdf'}
    )

@app.route('/add_notes', methods=['POST'])
@login_required
def add_notes():
    try:
        data = request.get_json()
        prediction_id = data.get('prediction_id')
        new_notes_text = data.get('notes')
        
        prediction = Prediction.query.get_or_404(prediction_id)
        
        # Ensure user owns this prediction
        if prediction.user_id != current_user.id:
            return {'error': 'Unauthorized'}, 403
        
        # Load existing notes
        existing_notes = json.loads(prediction.notes) if prediction.notes else {}
        
        # Update only the text field while preserving other data
        existing_notes['text'] = new_notes_text
        
        # Save merged notes back to prediction
        prediction.notes = json.dumps(existing_notes)
        db.session.commit()
        
        return {'success': True}
    except Exception as e:
        print(f"Error in add_notes: {str(e)}")  # Add debug logging
        return {'error': f'Error saving notes: {str(e)}'}, 500

@app.template_filter('from_json')
def from_json(value):
    """Parse JSON string in templates"""
    if not value:
        return {}
    try:
        return json.loads(value)
    except:
        return {}

@app.template_filter('month_is_current')
def month_is_current(date):
    """Check if a date is in the current month (dates are already in IST)"""
    # For testing/demo purposes, consider March 2025 as current month
    target_year = 2025
    target_month = 4
    
    # Convert to date object if it's a datetime
    date = date.date() if isinstance(date, datetime) else date
    
    # Debug information
    print(f"Input date (IST): {date}")
    print(f"Target: {target_year}-{target_month:02d}")
    print(f"Comparison: year={date.year == target_year}, month={date.month == target_month}")
    
    matches = date.year == target_year and date.month == target_month
    print(f"Final result: {matches}")
    return matches

@app.template_filter('avg')
def avg_filter(lst):
    """Calculate average of a list of numbers"""
    try:
        return sum(lst) / len(lst) if lst else 0
    except (TypeError, ZeroDivisionError):
        return 0

def migrate_to_ist():
    """Migrate existing predictions to IST timezone"""
    with app.app_context():
        predictions = Prediction.query.all()
        for prediction in predictions:
            # Convert UTC to IST (+5:30)
            ist_time = prediction.created_at + timedelta(hours=5, minutes=30)
            prediction.created_at = ist_time
        db.session.commit()
        print("Migration completed: All predictions updated to IST")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        migrate_to_ist()  # Run migration
    app.run(host='0.0.0.0', port=8000, debug=True) 