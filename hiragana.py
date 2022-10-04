import re
from pykakasi import kakasi

boin_dict = {
    "A": ["あ", "か", "さ", "た", "な", "は", "ま", "や", "ら", "わ"],
    "I": ["い", "き", "し", "ち", "に", "ひ", "み", "り"],
    "U": ["う", "く", "す", "つ", "ぬ", "ふ", "む", "ゆ", "る"],
    "E": ["え", "け", "せ", "て", "ね", "へ", "め", "れ"],
    "O": ["お", "こ", "そ", "と", "の", "ほ", "も", "よ", "ろ", "を"],
    "K": ["。"]
}

# オブジェクトをインスタンス化
kakasi = kakasi()
# モードの設定：J(Kanji) to H(Hiragana)
kakasi.setMode('J', 'H') 

# 変換して出力
conv = kakasi.getConverter()

answer = "おはようございます。今日もいい天気ですね。"

text = conv.do(answer)



# ひらがなの抽出
# hiragana = re.findall("[ぁ-ん]", text)
hiragana = [a for a in text]
print(hiragana)

note_list = []

for h in hiragana:
    for key, value in boin_dict.items():
        if h in value:
            note_list.append(key)


print(note_list)
