from django.urls import path
from . import views


urlpatterns = [
    path('', views.post_list, name='post_list'),
    path('post/<slug:slug>/', views.post_detail, name='post_detail'),
    path('post/new/', views.post_new, name='post_new'),
    path('post/<slug:slug>/edit/', views.post_edit, name='post_edit'),
    path('post/<int:pk>/delete/', views.post_delete, name='post_delete'),
    path('check_columns/', views.check_columns, name='check_columns'),
    path('landing/', views.landing_page, name='landing_page'),
    path('submit/', views.submit_landing_page, name='submit_landing_page'),
    path('post/<slug:slug>/edit/', views.post_edit, name='post_edit'),
    path('post/<pk>/delete/', views.post_delete, name='post_delete'),
    path('download_zip/', views.download_zip, name='download_zip'),
]