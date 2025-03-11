# Bone Age Prediction Web Application

This is a Flask web application that uses a deep learning model to predict bone age from hand X-ray images. The application includes user authentication and a modern UI built with Tailwind CSS.

## Features

- User authentication (signup/login)
- Secure file upload
- Real-time bone age prediction
- Modern and responsive UI with Tailwind CSS
- Support for various image formats

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd bone-age-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Create an account or login if you already have one.

4. Upload a hand X-ray image and get the predicted bone age.

## Model Information

The application uses a pre-trained deep learning model (`best_model.h5`) that was trained on a dataset of hand X-ray images. The model takes a grayscale image as input and predicts the bone age in years.

## File Structure

```
bone-age-prediction/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── best_model.h5         # Pre-trained model
└── templates/            # HTML templates
    ├── base.html         # Base template with common elements
    ├── index.html        # Landing page
    ├── login.html        # Login page
    ├── signup.html       # Registration page
    └── dashboard.html    # Main application interface
```

## Security Notes

- User passwords are hashed before storage
- File uploads are validated for type and size
- User authentication is required for predictions
- Session management is handled securely

## Contributing

Feel free to submit issues and enhancement requests! 