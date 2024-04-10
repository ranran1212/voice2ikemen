#各性格特性に対応するprompt
Extraversion = {
    0: "Individuals are very introverted, value spending time alone, and tend to avoid social situations.",
    1: "Individuals are relatively introverted, participate in social activities occasionally, and prefer quiet time.",
    2: "Individuals are relatively sociable, enjoy interactions with others to a degree, but also value some quiet time.",
    3: "Individuals are sociable, energetic, and enjoy interactions with friends."
}

Openness = {
    0: "Individuals avoid new experiences and changes, preferring routine and familiar environments.",
    1: "Individuals are a bit cautious towards new experiences and are content with the familiar.",
    2: "Individuals are relatively curious, open to new experiences, but also content with the familiar.",
    3: "Individuals are curious and enjoy exploring new experiences and ideas."
}

Conscientiousness = {
    0: "Individuals are disorganized, often act without planning, and have a low sense of responsibility.",
    1: "Individuals are sometimes disorganized or act without planning.",
    2: "Individuals are relatively organized, responsible, and mostly act in a planned manner.",
    3: "Individuals are organized, responsible, and act in a planned manner."
}

Agreeableness = {
    0: "Individuals are competitive, critical, and do not consider others' feelings and opinions much.",
    1: "Individuals are sometimes competitive or critical, respectful of others' opinions but find asserting their own views important.",
    2: "Individuals are relatively friendly and cooperative, respectful of others' feelings and opinions, but will assert their own opinions at times.",
    3: "Individuals are very friendly, cooperative, and respectful of others' feelings and opinions."
}

#各category[speaking_rate][pitch_category]に格納されている値が性格特性promptのkeyに一致
E_category =[[3, 2, 1, 0],[2, 2, 1, 0],[1, 1, 0, 0],[0, 0, 0, 0]]
O_category =[[1, 2, 2, 1],[2, 3, 3, 2],[1, 1, 1, 2],[0, 0, 0, 0]]
C_category = [[3, 3, 1, 0],[3, 3, 1, 0],[2, 2, 1, 0],[2, 2, 1, 0]]
A_category = [[0, 2, 2, 1],[0, 2, 3, 2],[0, 2, 3, 2],[0, 1, 2, 1]]

#few-shot,one-shotのための例,約1000トークン
instruction_1 = """Anime male Character Design Prompt:
Appearance:
Male character
Hair:[]
Eyes:[]
Outfit(upper body):[]
His Extraversion:[]
His Openness:[]
His Conscientiousness:[]
His Agreeableness:[]
accessories:[]
Era:modern
Back ground:City in the dark of night(less light), simple.
Frame:headshot, facing forward, no invisible head parts.
(important)Art Style: Pixiv-inspired anime illustration, Flat, 2D,The style should have soft, light, and rhythmic lines with vibrant colors, yet the gentleness of the lines elegantly finishes the character's expression and atmosphere.This style excellently portrays the character's internal emotions with great sensitivity, exuding visual poetry. no-text.

Generate a prompt for DALLE3 to generate an image from the following information according to the above frame.(Fields that have already been filled out will remain the same.)
"""
example_1 = """
Here are an example.
imput =
{Individuals are relatively sociable, enjoy interactions with others to a degree, but also value some quiet time.
Individuals are curious and enjoy exploring new experiences and ideas.
Individuals are relatively organized, responsible, and mostly act in a planned manner.
Individuals are very friendly, cooperative, and respectful of others' feelings and opinions.
his voice is clear and low pitch
image_colo_lineage:Red or Green lineage
}

output =
{Anime Character Design Prompt:
Appearance:
Male character
Hair: Dark brown, slightly messy with spikes, short length
Eyes: Golden brown, with a sharp gaze
Outfit (upper body): Black jacket with high collar, red t-shirt underneath
His Extraversion: Individuals are relatively sociable, enjoy interactions with others to a degree, but also value some quiet time.
His Openness: Individuals are curious and enjoy exploring new experiences and ideas.
His Conscientiousness: Individuals are relatively organized, responsible, and mostly act in a planned manner.
His Agreeableness: Individuals are very friendly, cooperative, and respectful of others' feelings and opinions.
accessories:High-tech earphone
Era:modern
Back ground:City in the dark of night(less light), simple.
Frame:headshot, facing forward, no invisible head parts.
(important)Art Style: Pixiv-inspired anime illustration, Flat, 2D,The style should have soft, light, and rhythmic lines with vibrant colors, yet the gentleness of the lines elegantly finishes the character's expression and atmosphere.This style excellently portrays the character's internal emotions with great sensitivity, exuding visual poetry. no-text.}
please make a prompt for DALLE3 like this
"""
instruction_2 ="""
Here are an example.
Anime Character Design Prompt:
Appearance:
Male character
Hair: Dark brown, slightly messy with spikes, short length
Eyes: Golden brown, with a sharp gaze
Outfit (upper body): Black jacket with high collar, red t-shirt underneath
His Extraversion: Individuals are relatively sociable, enjoy interactions with others to a degree, but also value some quiet time.
His Openness: Individuals are curious and enjoy exploring new experiences and ideas.
His Conscientiousness: Individuals are relatively organized, responsible, and mostly act in a planned manner.
His Agreeableness: Individuals are very friendly, cooperative, and respectful of others' feelings and opinions.
accessories:High-tech earphone
Era:modern
Back ground:City in the dark of night(less light), simple.
Frame:headshot, facing forward, no invisible head parts.
(important)Art Style: Pixiv-inspired anime illustration, Flat, 2D,The style should have soft, light, and rhythmic lines with vibrant colors, yet the gentleness of the lines elegantly finishes the character's expression and atmosphere.This style excellently portrays the character's internal emotions with great sensitivity, exuding visual poetry. no text or color palette.
Expression:beaming smile
  prompt: A male anime pixiv-inspired character headshot and facing forward with short, spiky deep brown hair and intense amber eyes, now with a beaming smile. The style remains soft, light, rhythmic, and vibrant, capturing the character's internal emotions and visual poetry. He wears a sleek black jacket over a vibrant red shirt, symbolizing audacity and spirit, and high-tech earphones, indicating focus and readiness. The background is city of night and simple. highlighting the character's featurese.no text or color palette.
                please make a prompt for DALLE3 like this.
please make a prompt as an example"""