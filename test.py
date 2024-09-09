import pytest
from app import app
import base64

# Create a test client using the Flask application configured for testing
@pytest.fixture
def client():
    app.config['TESTING'] = True
    return app.test_client()

# Test the home page
def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'AI Math Tutor' in response.data

# Test solving a problem
def test_solve(client):
    response = client.post('/solve', data={
        'problem': '2*x + 3 - 7',
        'operation': 'simplify'
    })
    assert response.status_code == 200
    assert b'Simplified Expression:' in response.data



def test_plot(client):
    response = client.post('/solve', data={
        'problem': 'x**2',
        'operation': 'plot'
    })
    assert response.status_code == 200

    # Decode the response data to a string
    response_data = response.data.decode('utf-8')

    # Look for the base64-encoded image pattern in the response data
    assert 'data:image/png;base64,' in response_data

    # Optional: Verify if the base64 string is a valid image
    img_data = response_data.split('data:image/png;base64,')[1]
    try:
        base64.b64decode(img_data)
        is_valid_image = True
    except Exception:
        is_valid_image = False

    assert is_valid_image




# Test feedback submission
def test_feedback(client):
    response = client.post('/feedback', data={
        'rating': '5',
        'comments': 'Great app!'
    })
    assert response.status_code == 200
    assert b'Thank you for your feedback!' in response.data
