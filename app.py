import json

import pandas as pd
from anytree import Node, RenderTree
from flask import Flask, render_template, request

from processing import translate_style, shuffle_inside_group

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key"
reflector = "female_classic_no_dress"

df = pd.read_pickle("./hidden_data/output_clean_4.pkl")
df_cats = pd.read_pickle("./hidden_data/df_cats.pkl")


with open("./hidden_data/style_list_240214.json", "r") as file:
    style_json = json.load(file)


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


def build_cats_view(style_json, df_cats):
    df_st = pd.DataFrame(columns=["gender", "style", "items"])
    dfs = []
    for gender, styles in style_json.items():
        for style, types in styles.items():
            for type, value in types.items():
                df_st = pd.DataFrame(
                    {"gender": gender, "style": style, "type": type, "value": value}
                )
                dfs.append(df_st)

    df_st = pd.concat(dfs, ignore_index=True)
    df_st["ru_name"] = df_st.apply(
        lambda x: translate_style(x["gender"] + "_" + x["style"]), axis=1
    )

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
        df_st,
        df_cats.set_index("category_code")[["path"]],
        how="left",
        left_on="value",
        right_index=True,
    )
    return df_cats_view


df_cats_view = build_cats_view(style_json, df_cats)
df_selected = df.copy()


@app.route("/", methods=["GET", "POST"])
def index():
    global df
    global df_selected
    if request.method == "POST":
        gender = request.form.get("gender")
        style = request.form.get("style")
        tl_i, tr_i, bl_i, br_i = 0, 0, 0, 0

    else:
        tl_i = int(request.args.get("tl_i", 0))
        tr_i = int(request.args.get("tr_i", 0))
        bl_i = int(request.args.get("bl_i", 0))
        br_i = int(request.args.get("br_i", 0))
        gender = request.args.get("gender", "female")
        style = request.args.get("style", "sport")

    print(gender, style, tl_i, tr_i, bl_i, br_i)
    col_selector = f"{gender}_{style}_pos"
    if request.method == "POST":
        df_selected = shuffle_inside_group(
            df.loc[~df[col_selector].isna()], col_selector
        )

    tl_item = df_selected.loc[df_selected[col_selector] == "top_left"].iloc[[tl_i]]
    tl_url = tl_item.photo_url.iloc[0]

    tr_item = df_selected.loc[df_selected[col_selector] == "top_right"].iloc[[tr_i]]
    tr_url = tr_item.photo_url.iloc[0]

    bl_item = df_selected.loc[df_selected[col_selector] == "bottom_left"].iloc[[bl_i]]
    bl_url = bl_item.photo_url.iloc[0]

    br_item = df_selected.loc[df_selected[col_selector] == "bottom_right"].iloc[[br_i]]
    br_url = br_item.photo_url.iloc[0]

    concat_table = pd.concat([tl_item, tr_item, bl_item, br_item]).reset_index(
        drop=True
    )
    concat_table["style"] = f"{gender}_{style}"
    print(concat_table.columns)
    columns_view = [
        "item_title",
        "brand",
        "color_base_title",
        "offer_price",
        "item_code",
        "category_4",
    ]
    look_table = (
        concat_table[columns_view]
        .rename(
            columns={
                "item_title": "Название",
                "brand": "Бренд",
                "color_base_title": "Базовый цвет",
                "offer_price": "Цена",
                "item_code": "Код товара",
                "category_4": "Категория",
            }
        )
        .to_html()
    )
    table_json = concat_table.dropna(axis=1).to_json(orient="records")

    category_tree = f"{style_tree(df_cats_view, f'{gender}_{style}')}Базовые цвета: {', '.join(style_json[gender][style]['color'])}"

    return render_template(
        "index.html",
        gender=gender,
        style=style,
        ru_name=translate_style(f"{gender}_{style}"),
        tl_url=tl_url,
        tl_i=tl_i,
        tr_url=tr_url,
        tr_i=tr_i,
        bl_url=bl_url,
        bl_i=bl_i,
        br_url=br_url,
        br_i=br_i,
        look_table=look_table,
        table_json=table_json,
        category_tree=category_tree,
    )


@app.route("/save-good-look", methods=["POST"])
def save_good_look():
    json_data = request.get_json()
    json_str = json.dumps(json_data).encode().decode("unicode_escape")
    with open("./hidden_data/review/looks_good.json", "a") as file:
        file.write(json_str + "\n")
    return "JSON data saved on the server"


@app.route("/save-bad-tl", methods=["POST"])
def save_bad_tl():
    json_data = request.get_json()
    json_str = json.dumps(json_data).encode().decode("unicode_escape")
    with open("./hidden_data/review/looks_bad_tl.json", "a+") as file:
        file.write(json_str + "\n")
    return "JSON data saved on the server"


@app.route("/save-bad-tr", methods=["POST"])
def save_bad_tr():
    json_data = request.get_json()
    json_str = json.dumps(json_data).encode().decode("unicode_escape")
    with open("./hidden_data/review/looks_bad_tr.json", "a+") as file:
        file.write(json_str + "\n")
    return "JSON data saved on the server"


@app.route("/save-bad-bl", methods=["POST"])
def save_bad_bl():
    json_data = request.get_json()
    json_str = json.dumps(json_data).encode().decode("unicode_escape")
    with open("./hidden_data/review/looks_bad_bl.json", "a+") as file:
        file.write(json_str + "\n")
    return "JSON data saved on the server"


@app.route("/save-bad-br", methods=["POST"])
def save_bad_br():
    json_data = request.get_json()
    json_str = json.dumps(json_data).encode().decode("unicode_escape")
    with open("./hidden_data/review/looks_bad_br.json", "a+") as file:
        file.write(json_str + "\n")
    return "JSON data saved on the server"
