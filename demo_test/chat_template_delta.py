from transformers import AutoTokenizer  
from verl.workers.rollout.schemas import (  
    AsyncRolloutRequest,   
    AsyncRolloutRequestStateEnum,   
    Message,   
    TokenizationSanityCheckModeEnum  
)  
from verl.tools.schemas import (  
    OpenAIFunctionToolCall,  
    OpenAIFunctionCallSchema,  
    OpenAIFunctionSchema,  
    OpenAIFunctionParametersSchema,  
    OpenAIFunctionPropertySchema,  
    OpenAIFunctionToolSchema  
)  
  
# 1. 加载 tokenizer  
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")  
  
# 2. 定义 tool schema（必须提供给 chat template）  
tool_schemas = [  
    OpenAIFunctionToolSchema(  
        type="function",  
        function=OpenAIFunctionSchema(  
            name="search",  
            description="Search the web",  
            parameters=OpenAIFunctionParametersSchema(  
                type="object",  
                properties={  
                    "query": OpenAIFunctionPropertySchema(  
                        type="string",  
                        description="Search query"  
                    )  
                },  
                required=["query"]  
            )  
        )  
    )  
]  
  
# 3. 准备包含 tool call 的消息列表  
messages = [  
    {"role": "system", "content": "You are a helpful assistant."},  
    {"role": "user", "content": "What's the weather in Beijing?"},  
]  
  
# 4. 创建 AsyncRolloutRequest  
req = AsyncRolloutRequest(  
    request_id="test-req",  
    state=AsyncRolloutRequestStateEnum.PENDING,  
    messages=[Message(**msg) for msg in messages],  
    multi_modal_data={"image": [], "video": []},  
    tool_schemas=tool_schemas,  # 提供 tool schemas  
    reward_scores={},  
    max_prompt_len=2048,  
    max_response_len=2048,  
    max_model_len=4096,  
    use_inference_chat_template=True,  
    tokenization_sanity_check_mode=TokenizationSanityCheckModeEnum.STRICT,  
    processing_class=tokenizer,  
)  
  
# 5. 模拟 assistant 生成带 tool call 的消息  
tool_calls = [  
    OpenAIFunctionToolCall(  
        id="call_1",  
        function=OpenAIFunctionCallSchema(  
            name="search",  
            arguments='{"query": "Beijing weather"}'  
        )  
    )  
]  
  
# 添加 assistant 消息（带 tool_calls）  
req.add_assistant_message(  
    tokenizer,   
    content="Let me search for that.",  
    tool_calls=tool_calls  
)  
  
# 6. 模拟 tool 返回结果  
tool_responses = ["The weather in Beijing is sunny, 25°C."]  
req.add_tool_response_messages(tokenizer, tool_responses)  
  
# 7. 添加最终的 assistant 回复  
req.add_assistant_message(  
    tokenizer,  
    content="The weather in Beijing is sunny with a temperature of 25°C."  
)  
  
# 8. 触发 sanity check  
req.finalize(tokenizer, reward_scores={})  
  
# 如果有 diff，会在日志中看到警告