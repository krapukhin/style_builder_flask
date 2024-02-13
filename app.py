import html
import json
from datetime import datetime

import pandas as pd
import requests
from anytree import Node, RenderTree
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import SelectField
from processing import translate_style

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key"
reflector = "female_classic_no_dress"

df = pd.read_pickle("./hidden_data/output_clean_3.pkl")


with open("./hidden_data/style_list_240126.json", "r") as file:
    style_json = json.load(file)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        gender = request.form.get("gender")
        style = request.form.get("style")
        print(gender, style)

        col_selector = f"{gender}_{style}_pos"
        # col_selector = f"female_sport_pos"

        tl_i = 0
        tl_item = df.loc[df[col_selector] == "top_left"].iloc[[tl_i]]
        tl_url = tl_item.photo_url.iloc[0]

        tr_i = 0
        tr_item = df.loc[df[col_selector] == "top_right"].iloc[[tr_i]]
        tr_url = tr_item.photo_url.iloc[0]

        bl_i = 0
        bl_item = df.loc[df[col_selector] == "bottom_left"].iloc[[bl_i]]
        bl_url = bl_item.photo_url.iloc[0]

        br_i = 0
        br_item = df.loc[df[col_selector] == "bottom_right"].iloc[[br_i]]
        br_url = br_item.photo_url.iloc[0]

        return render_template(
            "index.html",
            gender=gender,
            style=style,
            tl_url=tl_url,
            tl_i=tl_i,
            tr_url=tr_url,
            tr_i=tr_i,
            bl_url=bl_url,
            bl_i=bl_i,
            br_url=br_url,
            br_i=br_i,
        )
    else:
        tl_i = int(request.args.get("tl_i", 0))
        tr_i = int(request.args.get("tr_i", 0))
        bl_i = int(request.args.get("bl_i", 0))
        br_i = int(request.args.get("br_i", 0))
        gender = request.args.get("gender", "female")
        style = request.args.get("style", "sport")
        print(gender, style)

        col_selector = f"{gender}_{style}_pos"
        # col_selector = f"female_sport_pos"

        tl_item = df.loc[df[col_selector] == "top_left"].iloc[[tl_i]]
        tl_url = tl_item.photo_url.iloc[0]

        tr_item = df.loc[df[col_selector] == "top_right"].iloc[[tr_i]]
        tr_url = tr_item.photo_url.iloc[0]

        bl_item = df.loc[df[col_selector] == "bottom_left"].iloc[[bl_i]]
        bl_url = bl_item.photo_url.iloc[0]

        br_item = df.loc[df[col_selector] == "bottom_right"].iloc[[br_i]]
        br_url = br_item.photo_url.iloc[0]

        return render_template(
            "index.html",
            gender=gender,
            style=style,
            tl_url=tl_url,
            tl_i=tl_i,
            tr_url=tr_url,
            tr_i=tr_i,
            bl_url=bl_url,
            bl_i=bl_i,
            br_url=br_url,
            br_i=br_i,
        )
