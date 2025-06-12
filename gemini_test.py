import base64

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0.5,
#     google_api_key="xxxxxx",
#     # max_tokens=None,
#     # timeout=None,
#     # max_retries=2,
#     # other params...
# )

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="xxxxx")

vector = embeddings.embed_query("What are embeddings?")
print(vector[:5])

# Example using a public URL (remains the same)
# message_url = HumanMessage(
#     content=[
#         {
#             "type": "text",
#             "text": "Describe the image at the URL.",
#         },
#         {"type": "image_url", "image_url": "https://picsum.photos/seed/picsum/200/300"},
#     ]
# )
# result_url = llm.invoke([message_url])
# print(f"Response for URL image: {result_url.content}")

# Example using a local image file encoded in base64
# image_file_path = "/Users/ELI/Downloads/img-1.jpeg"

# with open(image_file_path, "rb") as image_file:
#     encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# message_local = HumanMessage(
#     content=[
#         {"type": "text", "text": "Extrae el texto de la imagen"},
#         {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_image}"},
#     ]
# )
# result_local = llm.invoke([message_local])
# print(f"Response for local image: {result_local.content}")