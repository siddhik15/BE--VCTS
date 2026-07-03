from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required  # <-- added
from django.contrib.auth.models import User
from django.contrib import messages
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import send_mail
from django.conf import settings
from .forms import ContactForm

# ================== MAIN PAGES ==================
def index(request):
    """Landing page view — shows the contact form and handles POST of the form."""
    form = ContactForm()
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('project_medical:contact_success')
    return render(request, 'project_medical/index.html', {'form': form})

def contact_success(request):
    """Simple success page after contact form is submitted."""
    return render(request, 'project_medical/contact_success.html')

def test_connection(request):
    return render(request, 'project_medical/test_connection.html')

# ================== PROTECTED FEATURES ==================
@login_required
def synthetic(request):
    """Render the synthetic patient generator page"""
    return render(request, 'project_medical/synthetic.html')

@login_required
def trial_simulation(request):
    """Render the trial simulation page"""
    return render(request, 'project_medical/trial_simulation.html')

@login_required
def predictive(request):
    """Render the predictive analytics page"""
    return render(request, 'project_medical/predictive.html')

@login_required
def accelerated(request):
    """Render the accelerated timelines page"""
    return render(request, 'project_medical/accelerated.html')

# ================== AUTHENTICATION ==================
def signup_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")

        # Password match check
        if password1 != password2:
            messages.error(request, "Passwords do not match.")
            return redirect("project_medical:signup")

        # Username already exists
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken.")
            return redirect("project_medical:signup")

        # Email already exists
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered.")
            return redirect("project_medical:signup")
        
        # Password length check
        if len(password1) < 6:
            messages.error(request, "Password must be at least 6 characters.")
            return redirect("project_medical:signup")

        # Create user
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password1
        )
        user.is_active = True  # Set active directly for now (no email verification)
        user.save()

        # Auto-login after signup
        login(request, user)
        messages.success(request, "Account created successfully!")
        return redirect("project_medical:index")

    return render(request, "project_medical/signup.html")

def activate_account(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except:
        user = None

    if user and default_token_generator.check_token(user, token):
        user.is_active = True
        user.save()
        return render(request, 'project_medical/activation_success.html')
    else:
        return render(request, 'project_medical/activation_invalid.html')

def login_view(request):
    # If user is already logged in, redirect to home
    if request.user.is_authenticated:
        return redirect('project_medical:index')
    
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')

        # Check if username exists
        if not User.objects.filter(username=username).exists():
            messages.error(request, "Username does not exist. Please sign up.")
            return redirect('project_medical:login')

        # Authenticate
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            # Check if there's a next URL to redirect to
            next_url = request.GET.get('next')
            if next_url:
                return redirect(next_url)
            return redirect('project_medical:index')
        else:
            messages.error(request, "Incorrect password. Please try again.")

    return render(request, 'project_medical/login.html')

def logout_view(request):
    logout(request)
    return redirect('project_medical:index')