from django import forms

class UserForm(forms.Form):
    review = forms.CharField(widget=forms.Textarea(attrs={"class": "myfield"}))
