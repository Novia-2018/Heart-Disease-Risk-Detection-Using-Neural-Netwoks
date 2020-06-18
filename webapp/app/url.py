from django.urls import path

from . import views

app_name = 'app'
urlpatterns = [
    path('',views.home,name='home'),
    path('quality/',views.quality,name='quality'),
    path('forms/',views.forms,name='forms'),
]
