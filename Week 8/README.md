# 디버깅 툴(Debugging) 정해서 run llms locally
---
Week8 정리본 : https://chanrankim.notion.site/8-cfd51bf4fd3e4ab386fbc03ab712c06e

---
---

# 1. 디버깅

llm으로 빌드하는 경우 원치 않은 출력이 생성되었는 지 명확하지 않기 때문에 디버깅이 필요하다.

## 추적

LangSmith와 같은 추적 기능이 있는 플랫폼은 디버깅을 위한 솔루션

LangSmith는 유료이고, LangFuse와 같은 무료 라이브러리 존재

## set_debug & set_verbose

Jupyter Notebook에서 프로토타입을 제작하거나 Python 스크립트를 실행하는 경우 Chain 실행의 중간 단계를 인쇄해 두면 도움이 된다.

간단한 에이전트를 만들어서, 디버깅 유무에 따른 출력 차이를 확인해보자

```python
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = load_tools(["ddg-search", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

agent.run("Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?")

'The director of the 2023 film Oppenheimer is Christopher Nolan and he is approximately 19345 days old in 2023.'
```

### set_debug(True)

글로벌 debug 플래그를 설정하면 콜백 지원(체인, 모델, 에이전트, 도구, 검색기)이 있는 모든 LangChain 구성 요소가 수신한 입력과 생성한 출력을 인쇄한다.

가장 자세한 설정이며 원시 입력과 출력을 모두 기록한다.

```python
from langchain.globals import set_debug

set_debug(True)

agent.run("Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?")
```

### set_verbose(True)

플래그를 설정하면 `verbose`입력 및 출력이 좀 더 읽기 쉬운 형식으로 출력된다.

특정 원시 출력(LLM 호출에 대한 토큰 사용 통계 등)에 대한 로깅을 건너뛰므로 애플리케이션 로직에 집중할 수 있다.

```python
from langchain.globals import set_verbose

set_verbose(True)

agent.run("Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?")
```

### Chain(…, verbose=True)

단일 객체에 대한 자세한 정보 범위를 지정할 수도 있는데, 이 경우 해당 객체에 대한 입력과 출력만(해당 객체에서 특별히 호출된 추가 콜백과 함께) 인쇄된다.

```python
# Passing verbose=True to initialize_agent will pass that along to the AgentExecutor (which is a Chain).
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent.run("Who directed the 2023 film Oppenheimer and what is their age? What is their age in days (assume 365 days per year)?")
```

## 기타 콜백

이상으로 기타 콜백 함수를 사용하여 구성 요소의 중간 단계를 기록하는 콜백을 사용할 수 있다. 

또는 사용자 지정 기능을 실행하기 위해 고유한 콜백을 구현할 수 있다.

---

# 2. 로컬에서 LLM 실행

## 사용 이유

1. Privacy : 데이터는 제3자에게 전송되지 않으며, 상업 서비스의 이용 약관에 따르지 않는다.
2. Cost : 토큰 집약적 애플리케이션에 중요한 추론 수수료가 없다.

## 개요

LLM을 로컬에서 실행할려면 아래와 같은 요구사항이 필요

1. Open-source LLM : 자유롭게 수정 및 공유할 수 있는 오픈소스 LLM
2. Inference : 허용 가능한 지연 시간으로 장치에서 해당 LLM을 실행할 수 있는 기능

### LLM 오픈소스

LLM은 아래와 같은 두 가지 차원에서 평가될 수 있다.

1. Base model : 기본 모델 종류, 훈련 방식
2. Fine-tuning approach : 파인튜닝되었는지 여부와 instruct 훈련인지 여부

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/333f96cf-396d-45ff-8331-232d41bd4d55/30d254b3-4de4-49d8-a87a-4ee1edec1f94/Untitled.png)

### 추론

오픈소스 LLM의 추론을 지원하기 위한 몇 가지 프레임워크

1. llama.cpp : 가중치 최적화/양자화를 사용한 llmam 추론 코드의 C++ 구현
2. gpt4all : 추론을 위한 최적화된 C 백엔드
3. Ollama : 장치에서 실행되고 LLM을 제공하는 앱에 모델 가중치와 환경을 번들로 묶음
4. llamafile : 모델 가중치와 모델을 실행하는 데 필요한 모든 것을 단일 파일에 묶어 추가 설치 단계 없이 이 파일에서 로컬로 LLM을 실행

일반적으로 위의 프레임워크는 몇 가지 작업을 수행

1. Quantization : 원시 모델 가중치의 메모리 사용량을 줄임
2. Efficient implementation for inference : 소비자 하드웨어에 대한 추론 지원

## 빠른 시작

ollama 테스트

```python
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = Ollama(
    model="llama2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)
llm.invoke("The first man on the moon was ...")
```

### LLMs

양자화된 모델 가중치에 접근하는 방법은 다양한데, 대표적으로 아래와 같다.

1. Huggingface : 많은 양자화된 모델을 다운로드할 수 있으며 llama.cpp와 같은 프레임워크로 실행
2. gpt4all : 다운로드 가능한 메트릭 및 관련 양자화 모델의 리더보드를 제공
3. Ollama : 여러 모델에 직접 액세스(pull)

### Ollama

Ollama를 사용하여 다음을 통해 모델을 가져옵니다 `ollama pull <model family>:<tag>`.

```python
from langchain_community.llms import Ollama

llm = Ollama(model="llama2:13b")
llm.invoke("The first man on the moon was ... think step by step")
```

### llama.cpp

llama.cpp는 다양한 모델과 호환된다.

`n_gpu_layers`: GPU 메모리에 로드할 레이어 수

- 값: 1
- 의미: 모델의 한 레이어만 GPU 메모리에 로드(1개면 충분한 경우가 많습니다).

`n_batch`: 모델이 병렬로 처리해야 하는 토큰 수

- 값: n_batch
- 의미: 1과 n_ctx 사이의 값을 선택하는 것이 좋습니다(이 경우 2048로 설정됨)

`n_ctx`: 토큰 컨텍스트 창

- 값: 2048
- 의미: 모델은 한 번에 2048개의 토큰 창을 고려합니다.

`f16_kv`: 모델이 키/값 캐시에 대해 반정밀도를 사용해야 하는지 여부

- 값: 참
- 의미: 이 모델은 메모리 효율성이 더 높은 반정밀도를 사용. Metal은 True만 지원

```python
%env CMAKE_ARGS="-DLLAMA_METAL=on"
%env FORCE_CMAKE=1
%pip install --upgrade --quiet  llama-cpp-python --no-cache-dirclear

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

llm.invoke("The first man on the moon was ... Let's think step by step")
```

### GPT4ALL

모델 탐색기 에서 다운로드한 모델 가중치를 사용

```python
from langchain_community.llms import GPT4All

llm = GPT4All(
    model="/Users/rlm/Desktop/Code/gpt4all/models/nous-hermes-13b.ggmlv3.q4_0.bin"
)

llm.invoke("The first man on the moon was ... Let's think step by step")
```

### llamafile

llamafiles는 모델 가중치와 특별히 컴파일된 버전을 단일 파일로 묶어 대부분의 컴퓨터에서 실행될 수 있는 모든 추가 종속성을 제공

또한 모델과 상호 작용하기 위한 API를 llama.cpp 제공하는 내장 추론 서버와 함께 제공

```python
# Download a llamafile from HuggingFace
wget https://huggingface.co/jartine/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile

# Make the file executable. On Windows, instead just rename the file to end in ".exe".
chmod +x TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile

# Start the model server. Listens at http://localhost:8080 by default.
./TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile --server --nobrowser

from langchain_community.llms.llamafile import Llamafile

llm = Llamafile()

llm.invoke("The first man on the moon was ... Let's think step by step.")
```

## 프롬프트

llama는 `ConditionalPromptSelector`모델 유형에 따라 프롬프트를 설정하는 데 사용

```python
# Set our LLM
llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
    n_gpu_layers=1,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)
```

```python
from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain_core.prompts import PromptTemplate

DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>> \n You are an assistant tasked with improving Google search \
results. \n <</SYS>> \n\n [INST] Generate THREE Google search queries that \
are similar to this question. The output should be a numbered list of questions \
and each should have a question mark at the end: \n\n {question} [/INST]""",
)

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an assistant tasked with improving Google search \
results. Generate THREE Google search queries that are similar to \
this question. The output should be a numbered list of questions and each \
should have a question mark at the end: {question}""",
)

QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=DEFAULT_SEARCH_PROMPT,
    conditionals=[(lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)],
)

prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)
prompt

PromptTemplate(input_variables=['question'], output_parser=None, partial_variables={}, template='<<SYS>> \n You are an assistant tasked with improving Google search results. \n <</SYS>> \n\n [INST] Generate THREE Google search queries that are similar to this question. The output should be a numbered list of questions and each should have a question mark at the end: \n\n {question} [/INST]', template_format='f-string', validate_template=True)
```

```python
# Chain
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What NFL team won the Super Bowl in the year that Justin Bieber was born?"
llm_chain.run({"question": question})

Sure! Here are three similar search queries with a question mark at the end:

1. Which NBA team did LeBron James lead to a championship in the year he was drafted?
2. Who won the Grammy Awards for Best New Artist and Best Female Pop Vocal Performance in the same year that Lady Gaga was born?
3. What MLB team did Babe Ruth play for when he hit 60 home runs in a single season?
``````output

llama_print_timings:        load time = 14943.19 ms
llama_print_timings:      sample time =    72.93 ms /   101 runs   (    0.72 ms per token,  1384.87 tokens per second)
llama_print_timings: prompt eval time = 14942.95 ms /    93 tokens (  160.68 ms per token,     6.22 tokens per second)
llama_print_timings:        eval time =  3430.85 ms /   100 runs   (   34.31 ms per token,    29.15 tokens per second)
llama_print_timings:       total time = 18578.26 ms
```

---

# 3. Pydantic 호환성

- Pydantic v2는 2023년 6월에 출시되었습니다( https://docs.pydantic.dev/2.0/blog/pydantic-v2-final/
    
    )
    
- v2에는 여러 가지 중요한 변경 사항이 포함되어 있습니다( https://docs.pydantic.dev/2.0/migration/
    
    )
    
- Pydantic v2와 v1은 동일한 패키지 이름에 있으므로 두 버전을 동시에 설치할 수 없습니다.

## Langchain Pydantic 마이그레이션

현재 `langchain>=0.0.267`LangChain에서는 사용자가 Pydantic V1 또는 V2를 설치할 수 있도록 허용합니다.

- [LangChain은 내부적으로 V1을](https://docs.pydantic.dev/latest/migration/#continue-using-pydantic-v1-features) 계속 사용할 것입니다 .
- 이 기간 동안 사용자는 pydantic 버전을 v1에 고정하여 변경 사항을 손상시키지 않거나 코드 전체에 pydantic v2를 사용하여 부분 마이그레이션을 시작할 수 있지만 LangChain의 경우 v1과 v2 코드를 섞지 않아도 됩니다.

사용자는 Pydantic v1에 고정한 후 LangChain이 내부적으로 v2로 마이그레이션하면 한꺼번에 코드를 업그레이드할 수도 있고, v2로 부분 마이그레이션을 시작할 수도 있지만 LangChain의 경우 v1과 v2 코드를 혼합해서는 안 됩니다.

## 정리

현재(2024-07-08) 기준으로 langchain 0.2.6 버전이고, pydantic v2를 대부분 사용하므로 pydantic v2를 사용할 것을 권장한다.

---
