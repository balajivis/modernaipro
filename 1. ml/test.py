# create a new file to call groq

import groq

groq.api_key = "gsk_1234567890"


def predict_home_price(data):
    return groq.predict(data)


print(predict_home_price("123 Main St, Anytown, USA"))

# new file to call openai
