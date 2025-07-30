import os
from langchain_openai import ChatOpenAI  # 或者根据您使用的模型选择正确的导入
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
# LLM模型
model = ChatOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key = os.getenv("QWQ_API_KEY"),
        model="qwq-plus-latest",
        streaming=True,
        temperature=0.5,
        top_p = 0.5,
        )

# prompt

prompt_template = ChatPromptTemplate.from_messages([
        ('system', '你是一个乐于助人的助理，用{language}回答问题'),
        # ('user', ''),
    ])

# chain
chain  = prompt_template | model 

# 聊天记录管理
class ChatData:
    def __init__(self):
         # 所有用户的聊天记录，都保存到这个字典中, key:session_id, value:chat_history
        self.chat_store = {}

    
    def get_session_history(self, session_id:str)->list:
        if session_id not in self.chat_store:
            self.chat_store[session_id] = ChatMessageHistory()
        return self.chat_store[session_id]

chatData = ChatData()

do_message = RunnableWithMessageHistory(
    chain,
    chatData.get_session_history,
    # 每次聊天发送的msg的key
    input_messages_key="my_input",
)

# 多轮对话

languages = ["中文","英文"]

config = {
    'configurable':{
        # 给当前会话定义session_id,如果不定义,则使用默认的session_id,默认的session_id为None
        'session_id':'njiter'
    }
}

if __name__ == '__main__':
    for lang in languages:
        for chunk in do_message.stream(
                {
                    "my_input":[HumanMessage(content="你好,我是南工程研究生")],
                    "language":lang
                },
                config = config
            ):
            print(chunk.content,end='',flush=True)
        print()
        print("*"*50)
        print()
    
    print(chatData.chat_store)