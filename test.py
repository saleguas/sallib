import os
from sallib.resources import gb
from sallib.ai.query import query_ai
from icecream import ic

print("Magick path:", gb("magick"))
def test_query_ai():
    test_image = os.path.abspath("test.png")
    ic(test_image)
    openai_query = query_ai(
        "what color is this image?", 
        model="gpt-4", 
        model_type="openai",
        image=test_image
    )
    ic(openai_query)
    print(openai_query)

test_query_ai()





