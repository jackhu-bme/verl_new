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

# original_template
# tokenizer.chat_template = "{% set image_count = namespace(value=0) %}\n{% set video_count = namespace(value=0) %}\n\n{% for message in messages %}\n  {% if loop.first and message.role != 'system' %}\n<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n  {% endif %}\n\n  {% if message.role == 'assistant' and message.tool_calls %}\n<|im_start|>assistant\n  {% if message.content is string %}\n{{ message.content }}\n  {% elif message.content %}\n{{ message.content }}\n  {% endif %}\n  {% for tool_call in message.tool_calls %}\n    {% if tool_call.function is defined %}\n      {% set tc = tool_call.function %}\n    {% else %}\n      {% set tc = tool_call %}\n    {% endif %}\n<tool_call>\n{\"name\": \"{{ tc.name }}\", \"arguments\": {{ tc.arguments | tojson }}}\n</tool_call>\n  {% endfor %}\n<|im_end|>\n\n  {% else %}\n<|im_start|>{{ message.role }}\n  {% if message.content is string %}\n{{ message.content }}<|im_end|>\n  {% else %}\n    {% for content in message.content %}\n      {% if content.type == 'image' or 'image' in content or 'image_url' in content %}\n        {% set image_count.value = image_count.value + 1 %}\n        {% if add_vision_id %}Picture {{ image_count.value }}:{% endif %}\n<|vision_start|><|image_pad|><|vision_end|>\n      {% elif content.type == 'video' or 'video' in content %}\n        {% set video_count.value = video_count.value + 1 %}\n        {% if add_vision_id %}Video {{ video_count.value }}:{% endif %}\n<|vision_start|><|video_pad|><|vision_end|>\n      {% elif 'text' in content %}\n{{ content.text }}\n      {% endif %}\n    {% endfor %}\n<|im_end|>\n  {% endif %}\n  {% endif %}\n\n{% endfor %}\n\n{% if add_generation_prompt %}\n<|im_start|>assistant\n{% endif %}"

# 使用修正后的 template  
tokenizer.chat_template = """{% set image_count = namespace(value=0) -%}  
{% set video_count = namespace(value=0) -%}  
{% for message in messages -%}  
{% if loop.first and message.role != 'system' -%}  
<|im_start|>system  
You are a helpful assistant.<|im_end|>  
{% endif -%}  
{% if message.role == 'assistant' and message.tool_calls -%}  
<|im_start|>assistant  
{% if message.content is string and message.content -%}  
{{ message.content }}  
{% endif -%}  
{% for tool_call in message.tool_calls -%}  
{% if tool_call.function is defined -%}  
{% set tc = tool_call.function -%}  
{% else -%}  
{% set tc = tool_call -%}  
{% endif -%}  
<tool_call>  
{"name": "{{ tc.name }}", "arguments": {{ tc.arguments | tojson }}}  
</tool_call>  
{% endfor -%}  
<|im_end|>  
{% else -%}  
<|im_start|>{{ message.role }}  
{% if message.content is string -%}  
{{ message.content }}<|im_end|>  
{% else -%}  
{% for content in message.content -%}  
{% if content.type == 'image' or 'image' in content or 'image_url' in content -%}  
{% set image_count.value = image_count.value + 1 -%}  
{% if add_vision_id -%}Picture {{ image_count.value }}:{% endif -%}  
<|vision_start|><|image_pad|><|vision_end|>  
{% elif content.type == 'video' or 'video' in content -%}  
{% set video_count.value = video_count.value + 1 -%}  
{% if add_vision_id -%}Video {{ video_count.value }}:{% endif -%}  
<|vision_start|><|video_pad|><|vision_end|>  
{% elif 'text' in content -%}  
{{ content.text }}  
{% endif -%}  
{% endfor -%}  
<|im_end|>  
{% endif -%}  
{% endif -%}  
{% endfor -%}  
{% if add_generation_prompt -%}  
<|im_start|>assistant  
{% endif -%}""" 



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
    use_inference_chat_template=False,  
    tokenization_sanity_check_mode=TokenizationSanityCheckModeEnum.STRICT,  
    processing_class=tokenizer,  
)  
  
# 5. 模拟 assistant 生成带 tool call 的消息  
tool_calls = [    
    OpenAIFunctionToolCall(    
        id="call_1",    
        function=OpenAIFunctionCallSchema(    
            name="search",    
            arguments={"query": "Beijing weather"}  # ✅ 修正:使用字典  
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