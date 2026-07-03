from django.urls import path
from . import views

app_name = 'project_medical'

urlpatterns = [
    # Home
    path('', views.index, name='index'),
    
    # Contact
    path('contact/success/', views.contact_success, name='contact_success'),
    
    # Authentication
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('activate/<uidb64>/<token>/', views.activate_account, name='activate'),
    
    # Features
    path('synthetic/', views.synthetic, name='synthetic'),
    path('trial-simulation/', views.trial_simulation, name='trial_simulation'),
    path('predictive/', views.predictive, name='predictive'),
    path('accelerated/', views.accelerated, name='accelerated'),
    
    # Utilities
    path('test-connection/', views.test_connection, name='test_connection'),
]