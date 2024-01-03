from flask import Flask
from wtforms.validators import DataRequired, Length
from flask_wtf import FlaskForm
from wtforms import TextAreaField

class InputForm(FlaskForm):
    """InputForm text
    """
    body = TextAreaField("Field for source of programming text", validators=[DataRequired(), Length(max=2048)])
        
