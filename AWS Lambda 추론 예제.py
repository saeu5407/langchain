import json
import re
import boto3

# 랭체인을 못써서 그냥 프롬포트 템플릿을 함수화
def make_prompt(index=None, text=None, question=None):
    prompt = f"""
    You are an AI language model assistant specialized in classifying topics. 
    You will be given a list of human input answers to a given input question. 
    Each answer is in the JSON format, with keys of \"index\" and \"answer\". 

    Your task is to identify if a given answer is NOT actually answering the given input question, based on its content, then output the index value.

    For example, for a given answer set  
    {{
        \"index\": \"answer1\",
        \"answer\": \"Amazon EC2 bare metal instances provide direct access to the 4th generation Intel Xeon Scalable processor and memory resources of the underlying server.\"
    }}

    If the \"answer\" is NOT answering the given input question, output the \"index\" value in a list as off_topic_answers: [\"answer1\"].
    If there are more than one answers that are not actually answering the given input question, output the all corresponding \"index\" values as a list.
    If all of the answers from the list of answers are on topic to answer the given input question, simply output as off_topic_answers: [\"-1\"].

    Here is the list of answer in JSON format:
    <answer_json>
    {{
        \"index\": {index},
        \"answer\": {text}
    }}
    </answer_json>

    Here is the input question:
    <input_question>
    {question}
    </input_question>

    REMEMBER: 
    output the index value of the answers that are NOT ON TOPIC ONLY, NOTHING ELSE! DO NOT OUTPUT EXPLANATION! ONLY THE LIST OF \"index\" values.
    """

    return prompt

# 단일 Question, Index, Text를 입력으로 받아서 인퍼런스 수행
def inference(question: str, index: str, text: str):

    prompt = make_prompt(question, index, text)

    client = boto3.client('bedrock-runtime')
    response = client.invoke_model(
        modelId='anthropic.claude-3-sonnet-20240229-v1:0',
        contentType='application/json',
        accept='application/json',
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ]
                }
            ]
        })
    )

    response_body = response['body'].read().decode('utf-8')
    response_payload = json.loads(response_body)
    answer = response_payload.get('content', 'No answer provided')
    answer = answer[0].get('text', 'No text')

    off_topic_pattern = r'off_topic_answers: \["(.*?)"\]'
    match = re.search(off_topic_pattern, answer)
    if match:
        off_topic_answer = match.group(1)
    else:
        off_topic_answer = 'No off topic answer found'

    decision = -1 if off_topic_answer == question else off_topic_answer

    return {
        'question': question,
        'index': index,
        'text': text,
        'decision': decision
    }

# 지금은 단일 값 기준으로 했지만 루프 돌리면 충분히 직렬로는 인퍼런스 가능할듯
def lambda_handler(event, context) -> str:

    question = event.get('question')
    index = event.get('index')
    text = event.get('text')

    result = inference(question, index, text)
    return result
