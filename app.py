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

output_clean = pd.read_pickle("./hidden_data/output_clean_3.pkl")


with open("./hidden_data/style_list_240126.json", "r") as file:
    style_json = json.load(file)


class MyForm(FlaskForm):
    gender = SelectField(
        "Выберите пол: ",
        choices=[("", "Select Gender"), ("male", "Мужской"), ("female", "Женский")],
    )
    selection = SelectField("Выберите стиль: ", choices=[])


def style_tree(df_cats_view, style=None):
    def category_tree_print(df):
        """Dataframe with a 'path' column which looks like 'Женское/Обувь/Мюли и Сабо'

        Returns:
            str: Tree view of dataframe
        """

        def build_tree(df):
            root = Node("Категории:")

            for index, row in df.iterrows():
                categories = row["path"].split("/")
                parent = root

                for category in categories:
                    if category != "NaN":
                        existing_child = next(
                            (
                                child
                                for child in parent.children
                                if child.name == category
                            ),
                            None,
                        )
                        if existing_child:
                            parent = existing_child
                        else:
                            node = Node(category, parent=parent)
                            parent = node

            return root

        def modify_tree_structure(node):
            if len(node.children) > 1:
                children = node.children
                node.children = tuple(children)

        def traverse_tree(node):
            for child in node.children:
                modify_tree_structure(child)
                traverse_tree(child)

        tree = build_tree(df)
        modify_tree_structure(tree)
        traverse_tree(tree)
        output = ""
        for pre, fill, node in RenderTree(tree):
            output += f"{pre}{node.name}\n"
        return output

    if style:
        g, s = style.split("_")[0], "_".join(style.split("_")[1:])
        df_cats_view = df_cats_view.loc[
            (df_cats_view["gender"] == g) & (df_cats_view["style"] == s)
        ]
    # print(df_cats_view.shape)
    return category_tree_print(df_cats_view.dropna(subset="path").loc[:, ["path"]])


@app.route("/save-bad-look", methods=["POST"])
def save_bad_look():
    json_data = request.get_json()
    json_str = json.dumps(json_data).encode().decode("unicode_escape")
    with open("./hidden_data/looks_bad.json", "a") as file:
        file.write(json_str + "\n")
    return "JSON data saved on the server"


@app.route("/save-good-look", methods=["POST"])
def save_good_look():
    json_data = request.get_json()
    json_str = json.dumps(json_data).encode().decode("unicode_escape")
    with open("./hidden_data/looks_good.json", "a") as file:
        file.write(json_str + "\n")
    return "JSON data saved on the server"


@app.route("/", methods=["GET", "POST"])
@app.route(
    "/<string:index1>/<string:index2>/<string:index3>/<string:index4>/<string:selector>"
)
def index(index1=0, index2=0, index3=0, index4=0, selector=None):
    index1, index2, index3, index4 = int(index1), int(index2), int(index3), int(index4)
    global reflector
    form = MyForm()
    if form.validate_on_submit():
        selected_gender = form.gender.data
        if selected_gender == "male":
            form.selection.choices = [
                ("female_sport_id", "Спортивный"),
                ("female_classic_no_dress_id", "Классический (без платьев)"),
                ("female_classic_dress_id", "Классический (с платьями)"),
                ("female_casual_no_dress_id", "Повседневный  (без платьев)"),
                ("female_casual_dress_id", "Повседневный (с платьями)"),
                ("female_romantic_no_dress_id", "Романтический  (без платьев)"),
                ("female_romantic_dress_id", "Романтический (с платьями)"),
                ("female_street_id", "Уличный"),
                ("female_business_id", "Бизнес"),
                ("female_grunge_id", "Гранж"),
                ("female_elegance_no_dress_id", "Элегантный  (без платьев)"),
                ("female_elegance_dress_id", "Элегантный (с платьями)"),
                ("female_boho_no_dress_id", "Бохо (без платьев)"),
                ("female_boho_dress_id", "Бохо (с платьями)"),
                ("female_fashion_street_id", "Модный стрит"),
                ("female_beach_no_dress_id", "Пляжный  (без платьев)"),
                ("female_beach_dress_id", "Пляжный (с платьями)"),
                ("female_vintage_no_dress_id", "Винтажный  (без платьев)"),
                ("female_vintage_dress_id", "Винтажный (с платьями)"),
                ("female_eco_no_dress_id", "Эко (без платьев)"),
                ("female_eco_dress_id", "Эко (с платьями)"),
                ("female_gothic_no_dress_id", "Готический (без платьев)"),
                ("female_gothic_dress_id", "Готический (с платьями)"),
                ("female_ethnic_no_dress_id", "Этнический (без платьев)"),
                ("female_ethnic_dress_id", "Этнический (с платьями)"),
            ]

        elif selected_gender == "female":
            form.selection.choices = [
                ("male_sport_id", "Спортивный"),
                ("male_casual_id", "Casual"),
                ("male_classic_id", "Классический"),
                ("male_street_id", "Уличный"),
                ("male_romantic_id", "Романтический"),
                ("male_st_style_id", "Стрит"),
                ("male_grunge_id", "Гранж"),
                ("male_elegance_id", "Элегантный"),
                ("male_punk_id", "Панк"),
                ("male_boho_id", "Бохо"),
            ]
    if form.selection.data is None:
        selector = reflector
        selector_name = translate_style(selector)
    else:
        selector = form.selection.data
        selector_name = translate_style(selector)
        reflector = form.selection.data

    style = selector.replace("_id", "")

    item1 = output_clean.loc[output_clean[style + "_pos"] == "top_left"].iloc[[index1]]
    item2 = output_clean.loc[output_clean[style + "_pos"] == "top_right"].iloc[[index2]]
    item3 = output_clean.loc[output_clean[style + "_pos"] == "bottom_left"].iloc[
        [index3]
    ]
    item4 = output_clean.loc[output_clean[style + "_pos"] == "bottom_right"].iloc[
        [index4]
    ]
    image1 = item1.photo_url.iloc[0]
    image2 = item2.photo_url.iloc[0]
    image3 = item3.photo_url.iloc[0]
    image4 = item4.photo_url.iloc[0]

    columns_view = [
        "item_title",
        "brand",
        "color_base_title",
        "offer_price",
        "item_code",
    ]  # color или color_base_title
    look_table = pd.concat([item1, item2, item3, item4]).reset_index(drop=True)
    table = look_table[columns_view].to_html()
    look_table["style"] = selector
    table_json = look_table.dropna(axis=1).to_json(orient="records")

    g, s = style.split("_")[0], "_".join(style.split("_")[1:])

    category_tree = f"{style_tree(df_cats_view, style)}Базовые цвета: {', '.join(style_json[g][s]['color'])}"

    return render_template(
        "index.html",
        image1=image1,
        image2=image2,
        image3=image3,
        image4=image4,
        index1=index1,
        index2=index2,
        index3=index3,
        index4=index4,
        selector=selector,
        selector_name=selector_name,
        table=table,
        table_json=table_json,
        category_tree=category_tree,
        form=form,
    )


df = pd.DataFrame(columns=["gender", "style", "items"])
dfs = []
for gender, styles in style_json.items():
    for style, types in styles.items():
        for type, value in types.items():
            df = pd.DataFrame(
                {"gender": gender, "style": style, "type": type, "value": value}
            )
            dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df["ru_name"] = df.apply(
    lambda x: translate_style(x["gender"] + "_" + x["style"]), axis=1
)

df_cats = pd.read_pickle("./hidden_data/df_cats.pkl")
df_cats["path"] = df_cats.apply(
    lambda x: "/".join(
        [
            c
            for c in [
                x["category_1"],
                x["category_2"],
                x["category_3"],
                x["category_4"],
            ]
            if c is not None
        ]
    ),
    axis=1,
)
df_cats_view = pd.merge(
    df,
    df_cats.set_index("category_code")[["path"]],
    how="left",
    left_on="value",
    right_index=True,
)
