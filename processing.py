import datetime
import html
import json
import re
from io import BytesIO

import pandas as pd
import requests
from IPython.display import HTML  # from IPython.core.display import HTML, display
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import create_engine

with open("./hidden_data/style_translate.json", "r") as file:
    style_translate = json.load(file)


def translate_style(name, style_translate=style_translate):
    name_split = name.replace("_id", "").split("_", maxsplit=1)
    gender = style_translate[name_split[0]]

    if gender == "Женский":
        if name_split[1].find("_dress") != -1:
            if name_split[1].find("_no_dress") != -1:
                dress = " (без платьев)"
            else:
                dress = " (с платьями)"
        else:
            dress = ""

    else:
        dress = ""
    style = name_split[1].replace("_no_dress", "").replace("_dress", "")
    return f"{gender} {style_translate[style]} стиль{dress}"


def style_list_df(style_json):
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
    return df


def json_log_to_df(path, variable_name="table_json"):
    with open(path) as f:
        data = f.readlines()
    frames = []
    for i in range(len(data)):
        single_frame = json.loads(data[i])[variable_name]
        single_frame = html.unescape(single_frame)
        single_frame = json.loads(single_frame)
        single_frame = pd.DataFrame(single_frame)
        single_frame["look_n"] = i + 1
        frames += [single_frame]

    look_index = pd.concat(frames).reset_index(drop=True)
    return look_index


def update_data_VPN():
    """Turn-on VPN to get data (~40 seconds)"""
    AS_engine = create_engine(f"")

    assortment = pd.read_sql("""select * from analytics.v$assortment""", AS_engine)
    assortment.to_pickle("hidden_data/assortment.pkl")

    # df_it = pd.read_sql("""select * from analytics.v$items""", AS_engine)
    # df_collit = pd.read_sql("""select * from analytics.v$collection_items""", AS_engine)
    df_cats = pd.read_sql("""select * from analytics.v$categories""", AS_engine)
    df_cats.to_pickle("hidden_data/df_cats.pkl")
    # df_color_concrete
    # df_brands = pd.read_sql("""select * from analytics.v$brands""", AS_engine)
    # df_sub_brands = pd.read_sql("""select * from analytics.v$sub_brands""", AS_engine)


def processing(assortment, df_cats):
    # df_color_base = pd.read_csv("data/color_base_231208.csv", index_col=0, low_memory=False)
    dict_color_base = {
        "1": "Синий",
        "2": "Красный",
        "3": "Серый",
        "4": "Белый",
        "5": "Коричневый",
        "6": "Чёрный",
        "7": "Бежевый",
        "8": "Зелёный",
        "9": "Фиолетовый",
        "10": "Жёлтый",
        "11": "Серебряный",
        "12": "Прозрачный",
        "13": "Розовый",
        "14": "Хаки",
        "15": "Золотой",
        "16": "Бесцветный",
        "17": "Разноцветный",
        "18": "Бронзовый",
        "19": "Чёрно-белый",
        "20": "Оранжевый",
        "21": "Леопардовый",
        "40": "Голубой",
        "41": "Кремовый",
        "42": "Бордовый",
        "43": "Сиреневый",
        "NC": "NO COLOR",
        "22": "Белое золото",
        "23": "Розовое золото",
        "24": "Чернёное белое золото",
        "25": "Жёлтое золото",
        "26": "Розовое и белое золото",
        "27": "Жёлтое и частично чернёное белое золото",
        "28": "Розовое золото и чернёное белое золото",
        "29": "Чернёное серебро с позолотой",
        "30": "Белое и чернёное золото",
        "31": "Белое золото и жёлтое золото",
        "32": "Белое золото и розовое золото",
        "33": "Белое и розовое золото",
        "34": "Розовое и жёлтое золото",
        "35": "Белое и жёлтое золото",
        "36": "Титан",
        "37": "Платина",
        "38": "Перламутровый",
        "39": "Серебро",
    }
    df_color_base = (
        pd.DataFrame.from_dict(
            dict_color_base, orient="index", columns=["color_base_title"]
        )
        .reset_index()
        .rename(columns={"index": "color_base_id"})
    )

    brand_styling = pd.read_csv("data/brand_styling_ChatGPT.csv").rename(
        columns={
            "brand": "brand_name",
            "style": "brand_style",
            "pricing": "brand_pricing",
        }
    )
    brand_styling_from_guide = pd.read_excel("data/styling_guide_brands.xlsx")
    brand_styling_from_guide["isPriority"].fillna(0, inplace=True)
    brand_styling_from_guide["isFashion"].fillna(0, inplace=True)
    brand_styling_from_guide["Name"] = brand_styling_from_guide["Name"].apply(
        lambda x: x.strip()
    )
    brand_styling_from_guide = brand_styling_from_guide.rename(
        columns={
            "Name": "brand_name",
            "type": "brand_pricing_guide",
            "isPriority": "isPriority_guide",
            "isFashion": "isFashion_guide",
        }
    )
    ## Processing columns
    # add color columns
    df_full = assortment.copy()

    print(f"All items: {df_full.shape}")
    df_full = df_full.loc[(df_full.quantity > 0) & (df_full.is_published == True)]
    print(f"Keep available items: {df_full.shape[0]} items")

    df_full["url"] = "https://collect.tsum.ru/item/" + df_full["item_code"]
    # add first material in description (column - main_composition)
    df_full["main_composition"] = df_full["composition"].str.extract(
        r"^(\w+)", expand=False
    )

    # add base-color name
    df_full = pd.merge(
        df_full,
        df_color_base,
        how="left",
        left_on="color_base_code",
        right_on="color_base_id",
    ).drop(columns=["color_base_id"])

    # add brand-classification (columns - brand_style, brand_pricing)
    df_full = pd.merge(
        df_full, brand_styling, how="left", left_on="brand", right_on="brand_name"
    ).drop(columns=["brand_name"])

    # add brand-info from guide (columns - brand_pricing_guide
    # isPriority_guide, isFashion_guide)
    df_full = pd.merge(
        df_full,
        brand_styling_from_guide,
        how="left",
        left_on="brand",
        right_on="brand_name",
    ).drop(columns=["brand_name"])

    drop_list = [df_full.loc[df_full.id == 140999893].index[0]]
    df_full = df_full.drop(index=drop_list)

    # magic for None in 'photo_url'-column with non-empty 'photos'-column (ITEM132110)
    df_full.loc[df_full.photo_url.isna(), "photo_url"] = df_full["photos"].apply(
        lambda x: [i for i in x if "public" in i.keys()][0]["public"]
    )
    print(
        f"Проверка, что ничего не потеряли: {df_full.shape[0]} + {len(drop_list)} items"
    )
    return df_full


def df_cleaner(df_full):
    ### Drop columns
    df_clean = df_full.drop(
        columns=[
            # "id",
            "stock_unit_code",
            "offer_code",
            "location_id",
            # "quantity",
            "terminate_reason",
            "available_from",  # ?какая то дата
            "seller_id",
            # "offer_price",
            "invoice_number",
            # "item_code",
            # "item_title",
            "item_description",
            # "brand",
            # "sub_brand",
            "external_data",  # словарь с информацией
            # "ru_size",
            "size",
            "condition_code",
            "condition",
            "defects",
            "defects_public",
            "color_code",
            "color_base_code",
            # "color",
            "image_count",
            # "tsum_price",
            "published_at",
            "item_created_at",
            "is_draft",
            # "is_published",
            "is_used",
            "measurement",
            "kit",
            # "category_1",
            # "category_1_code",
            # "category_2",
            # "category_2_code",
            # "category_3",
            # "category_3_code",
            # "category_4",
            # "category_4_code",
            "created_at",
            "updated_at",
            "ax_item_id",
            "ax_vendor_code",
            "ax_color_id",
            "ax_manufacturer_size",
            "ax_folder_name",
            "ax_season",
            "composition",
            "offer_created_at",
            "tsum_price_created_at",
            "category_code",
            "parent_category_code",
            "category_name",
            "vendor_code",
            "reserved_at",
            "availiable_for_purchase",
            # "sections",
            "genders",  # дублируется с sections
            "actual_at",
            "photos",
            # "photo_url",
            # "brand_collection_code",
            # "category_1_collection_code",
            # "category_2_collection_code",
            # "category_3_collection_code",
            # "category_4_collection_code",
            "currency_code",
            "defects_created_at",
            "defects_public_created_at",
            "composition_created_at",
            "model",
            "fabric",
            # "brand_style",
            # "brand_pricing",
            # "brand_pricing_guide",
            # "isPriority_guide",
            # "isFashion_guide",
            # "main_composition",
        ]
    )
    return df_clean


def get_item_img_url(df, debug=False):
    df = df.loc[df.score > 0]
    groups = df.groupby("score")
    for score, df_mini in list(groups)[::-1]:
        stopper = False
        df_sample = df_mini.sample(frac=1)  # Shuffle the dataframe
        for i in range(len(df_mini)):
            item = df_sample.iloc[[i]]
            item_url = item.iloc[0].url
            image_url = item.iloc[0].photo_url
            # image_url = get_image_url(item_url) # old
            if image_url is not None:
                if debug:
                    print(item_url)
                    print(image_url)
                # display(HTML(item.to_html(render_links=True, escape=False)))
                stopper = True
                return item, image_url
                # break
            else:
                if debug:
                    print(f"Bad item_url: {item_url}")
                continue
        if stopper:
            break

    return None


def score_items(
    df_all, cats, col_name, size=None, L_clr=None, L_style=None, L_price=None
):
    top_df = df_all.copy()
    top_df[col_name] = 0
    top_df = top_df.loc[top_df.category_4_code.isin(cats)]
    # top_df = top_df.loc[top_df.category_code.isin(cats)] # old
    if size:
        top_df.loc[top_df.ru_size.isin(size), col_name] += 1
    if not L_clr:
        L_clr = list(df_all.color_base_title.unique())
    top_df.loc[top_df.color_base_title.isin(L_clr), col_name] += 10
    if L_style:
        top_df.loc[top_df.brand_style == L_style, col_name] += 1
    if L_price:
        top_df.loc[top_df.brand_pricing == L_price, col_name] += 1
    top_df = top_df.sort_values(col_name, ascending=False)
    return top_df


def get_image_url(item_url):
    response = requests.get(item_url)
    if response.status_code == 200:
        html_content = response.text
        pattern = r"(https://preowned-cdn\.tsum\.com/sig/[a-f0-9]{32}/height/1526/document-hub/[a-zA-Z0-9]+\.(jpg|jpeg|png|gif))"
        matches = re.findall(pattern, html_content)
        return matches[0][0]
    else:
        return None


def get_img(df_it, cats, debug=False):
    for i in range(10):
        item = df_it.loc[df_it["category_code"].isin(cats)].sample(1)
        item_url = item.iloc[0].url
        image_url = get_image_url(item_url)
        if image_url is not None:
            break
        else:
            if debug:
                print(f"Bad item_url: {item_url}")
            time.sleep(1)
            continue

    if debug:
        print(item_url)
        print(image_url)
    display(item)
    return image_url


def merge_look_1x3(top_image, bottom_image, shoes_image):
    # Download the images
    response1 = requests.get(top_image)
    response2 = requests.get(bottom_image)
    response3 = requests.get(shoes_image)

    # Open the images
    img1 = Image.open(BytesIO(response1.content))
    img2 = Image.open(BytesIO(response2.content))
    img3 = Image.open(BytesIO(response3.content))

    # Resize the images to have the same width
    width, height = img1.size
    img2 = img2.resize((width, height))
    img3 = img3.resize((width, height))

    # Create a new blank image with the required size
    merged_image = Image.new("RGB", (width * 3, height))

    # Paste the images vertically
    merged_image.paste(img1, (0, 0))
    merged_image.paste(img2, (width, 0))
    merged_image.paste(img3, (width * 2, 0))

    # Show the merged image
    return merged_image


def merge_look_2x2(
    gender,
    style,
    top_left,
    bottom_left=None,
    bottom_right=None,
    top_right=None,
):
    # Download the images
    width = 247
    height = 381

    tl = single_image(top_left, (width, height))
    tr = single_image(top_right, (width, height))
    bl = single_image(bottom_left, (width, height))
    br = single_image(bottom_right, (width, height))

    # Create a new blank image with the required size
    merged_image = Image.new("RGB", (width * 2, height * 2))

    # Paste the images vertically
    merged_image.paste(tl, (0, 0))
    merged_image.paste(tr, (width, 0))
    merged_image.paste(bl, (0, height))
    merged_image.paste(br, (width, height))

    # Show the merged image
    text = f"{gender} {style}"
    draw = ImageDraw.Draw(merged_image)  # Create an ImageDraw object
    font = ImageFont.truetype("Arial", 40)  # Define the font and size for the title
    text_width, text_height = draw.textsize(text, font=font)
    title_position = (
        (width * 2 - text_width) // 2,
        10,
    )  # Define the position for the title
    draw.text(title_position, text, font=font, fill="black")
    merged_image.save(
        f'images/{datetime.datetime.now().strftime("%y%m%d-%H%M%S")}_{gender}_{style}.png'
    )

    return merged_image


def single_image(url, size=(247, 381)):
    if url:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = img.resize(size)

        return img
    else:
        return Image.new("RGB", size, "white")


def merge_look_3x2(
    gender,
    style,
    top_image,
    bottom_image=None,
    shoes_image=None,
    outerwear_image=None,
    bags_image=None,
):
    # Download the images
    width = 247
    height = 381

    img1 = single_image(top_image, (width, height))
    img2 = single_image(bags_image, (width, height))
    img3 = single_image(outerwear_image, (width, height))
    img4 = single_image(bottom_image, (width, height))
    img5 = single_image(shoes_image, (width, height))
    img6 = single_image(None, (width, height))

    # Create a new blank image with the required size
    merged_image = Image.new("RGB", (width * 3, height * 2))

    # Paste the images vertically
    merged_image.paste(img1, (0, 0))
    merged_image.paste(img2, (width, 0))
    merged_image.paste(img3, (width * 2, 0))
    merged_image.paste(img4, (0, height))
    merged_image.paste(img5, (width, height))
    merged_image.paste(img6, (width * 2, height))

    # Show the merged image
    text = style
    draw = ImageDraw.Draw(merged_image)  # Create an ImageDraw object
    font = ImageFont.truetype("Arial", 40)  # Define the font and size for the title
    text_width, text_height = draw.textsize(text, font=font)
    title_position = (
        (width * 2 - text_width) // 2,
        10,
    )  # Define the position for the title
    draw.text(title_position, text, font=font, fill="black")
    merged_image.save(
        f'images/{datetime.datetime.now().strftime("%y%m%d-%H%M%S")}_{gender}_{style}.png'
    )

    return merged_image


def get_item_img(df, debug=False):
    df = df.sample(frac=1).sort_values("score", ascending=False)
    for i in range(len(df)):
        item = df.iloc[[i]]
        item_url = item.iloc[0].url
        image_url = get_image_url(item_url)
        if image_url is not None:
            if debug:
                print(item_url)
                print(image_url)
            display(HTML(item.to_html(render_links=True, escape=False)))
            return image_url
        else:
            if debug:
                print(f"Bad item_url: {item_url}")
            continue
    return None


def shuffle_inside_group(df, group):
    output = df.copy()
    output = output.loc[
        output.photo_url.isna() == False
    ]  # df_look.loc[df_look.photo_url.isna() == False]
    output = (
        output.groupby(group, as_index=False)
        .apply(lambda x: x.sample(frac=1))
        .reset_index(drop=True)
    )  # shuffle inside group

    output = output.sort_index(ascending=False).reset_index(
        drop=True
    )  # sort by biggest score
    return output


def look_id(df, new_name):
    output = df.copy()
    output = output.reset_index().rename(columns={"index": new_name})  # assign look_id
    output[new_name] = output[new_name] + 1  # start from 1
    return output[output.columns[1:].to_list() + output.columns[:1].to_list()]


def df_equalizer(
    df_list, mode="multiplier"
):  # [pd.DataFrame({'col1': [1, 2, 3]}), pd.DataFrame({'col1': [2, 3,3, 4]}),pd.DataFrame({'col1': [3, 4, 5,6,7]})]
    if mode == "multiplier":
        n_rows = max([x.shape[0] for x in df_list])
    elif mode == "cutter":
        n_rows = min([x.shape[0] for x in df_list])
    output_list = []
    for df_i in df_list:
        output_list += [
            pd.concat(
                [df_i] * (n_rows // df_i.shape[0]) + [df_i.iloc[: n_rows % len(df_i)]],
                ignore_index=True,
            )
        ]
    print([x.shape[0] for x in output_list])
    return output_list


# unused
# import random

# ########## setup
# L_gender = "female"  # random.choice(["female", "male"])
# L_clothing_size = (
#     None  # random.choice([str(x) for x in clothing_sizes[L_gender].keys()])
# )
# L_clothing_size_RU = None  # clothing_sizes[L_gender][L_clothing_size]
# L_shoe_size = None  # random.choice([x for x in shoe_sizes[L_gender].keys()])
# L_shoe_size_RU = None  # shoe_sizes[L_gender][L_shoe_size]
# # L_clr = random.choice(list(df_clean.color_base_title))
# L_clr = list(
#     df_full.color_base_title.unique()
# )  # color_match[random.choice(list(color_match.keys())[:3])]
# L_style = None  # random.choice(list(df_clean.brand_style.dropna()))
# L_price = None  # random.choice(list(df_clean.brand_pricing.dropna()))
# ################

# print(
#     f"""Gender: {L_gender}
# Clothes size: {L_clothing_size} {L_clothing_size_RU}
# Shoe size: {L_shoe_size} {L_shoe_size_RU}
# Color: {L_clr}
# Style: {L_style}
# Price: {L_price}
# """
# )

# color_match = {
#     "bw": [
#         "Белый",
#         "Чёрно-белый",
#         "Чёрный",
#         "Прозрачный",  # черный
#     ],
#     "brown": [
#         "Коричневый",
#         "Синий",
#     ],
#     "neutral": [
#         "Бежевый",
#         "Кремовый",
#         "Перламутровый",
#         "Хаки",  # todo убрать из нейтральных?
#     ],
#     "silver": [
#         "Серый",  # todo убрать? цвет не очень металик
#         "Белое золото",
#         "Белое и жёлтое золото",
#         "Серебряный",
#         "Розовое и жёлтое золото",  # серебро или золото
#         "Чернёное серебро с позолотой",
#     ],
#     "bronze": [
#         "Бронзовый",
#         "Розовое золото",
#     ],
#     "gold": [
#         "Жёлтое золото",
#         "Жёлтое и частично чернёное белое золото",
#         "Золотой",
#     ],
#     "leopard": [
#         "Леопардовый",
#     ],
#     "bright": [
#         "Бордовый",
#         "Красный",
#         "Оранжевый",
#         "Жёлтый",
#         "Зелёный",
#         "Голубой",
#         "Фиолетовый",
#         "Сиреневый",
#         "Розовый",
#         "Разноцветный",
#     ],
#     "zero_items": [
#         "Бесцветный",  # 0 товаров
#         "Платина",  # 0 товаров
#         "Белое золото и жёлтое золото",  # 0 товаров
#         "Розовое золото и чернёное белое золото",  # 0 товаров
#         "Розовое и белое золото",  # 0 товаров
#         "Чернёное белое золото",  # 0 товаров
#     ],
# }


# cats = {
#     "male": {
#         "top_cats": [
#             "CAT-155",  #'Мужское/Одежда/Верхняя одежда'
#             "CAT-171",  #'Мужское/Одежда/Пиджаки'
#             "CAT-176",  #'Мужское/Одежда/Рубашки'
#             "CAT-180",  #'Мужское/Одежда/Свитеры и Кардиганы'
#             "CAT-186",  #'Мужское/Одежда/Свитшоты и Толстовки'
#             "CAT-199",  #'Мужское/Одежда/Футболки и Майки'
#         ],
#         "bottom_cats": [
#             "CAT-149",  #'Мужское/Одежда/Брюки'
#             "CAT-164",  #'Мужское/Одежда/Джинсы'
#         ],
#         "shoes_cats": [
#             "CAT-209",  #'Мужское/Обувь/Классическая обувь'
#             "CAT-215",  #'Мужское/Обувь/Летняя обувь'
#             "CAT-220",  #'Мужское/Обувь/Повседневная обувь'
#             "CAT-227",  #'Мужское/Обувь/Ботинки и Полусапоги'
#             "CAT-205",  #'Мужское/Обувь/Кеды и Кроссовки'
#         ],
#     },
#     "female": {
#         "top_cats": [
#             "CAT-12",  # 'Женское/Одежда/Блузы и Рубашки'
#             "CAT-21",  # 'Женское/Одежда/Верхняя одежда'
#             "CAT-50",  # 'Женское/Одежда/Свитеры и Кардиганы'
#             "CAT-56",  # 'Женское/Одежда/Свитшоты и Толстовки'
#             "CAT-66",  # 'Женское/Одежда/Футболки и Топы'
#         ],
#         "bottom_cats": [
#             "CAT-15",  # 'Женское/Одежда/Брюки'
#             "CAT-32",  # 'Женское/Одежда/Джинсы'
#             "CAT-77",  # 'Женское/Одежда/Юбки'
#         ],
#         "shoes_cats": [
#             "CAT-102",  # 'Женское/Обувь/Без каблука'
#             "CAT-109",  # 'Женское/Обувь/Босоножки'
#             "CAT-128",  # 'Женское/Обувь/Ботинки и Ботильоны'
#             "CAT-98",  # 'Женское/Обувь/Кеды и Кроссовки'
#             "CAT-112",  # 'Женское/Обувь/Мюли и Сабо'
#             "CAT-115",  # 'Женское/Обувь/Сапоги'
#             "CAT-120",  # 'Женское/Обувь/Туфли'
#         ],
#     },
#     "unused": [
#         "CAT-224",  #'Мужское/Обувь/Домашняя обувь'
#         "CAT-141",  #'Мужское/Одежда/Белье и Домашняя одежда'
#         "CAT-167",  #'Мужское/Одежда/Костюмы'
#         "CAT-196",  #'Мужское/Одежда/Пляжные принадлежности'
#         "CAT-190",  #'Мужское/Одежда/Спортивная одежда'
#         "CAT-154",  #'Мужское/Одежда/Шорты'
#         "CAT-125",  # 'Женское/Обувь/Домашняя обувь'
#         "CAT-40",  # 'Женское/Одежда/Платья'
#         "CAT-3",  # 'Женское/Одежда/Белье и Домашняя одежда'
#         "CAT-45",  # 'Женское/Одежда/Пляжная одежда'
#         "CAT-36",  # 'Женское/Одежда/Жакеты и Костюмы'
#         "CAT-60",  # 'Женское/Одежда/Спортивная одежда'
#         "CAT-74",  # 'Женское/Одежда/Шорты'
#     ],
# }
