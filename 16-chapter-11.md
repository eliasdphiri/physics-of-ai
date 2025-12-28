Chapter 11: Agentic Architectures: Loops, Memory, and ToolsIn This ChapterThe Anatomy of an Autonomous AgentThe ReAct Loop: Reason, Act, ObserveExtending the Action Space: Giving AI Hands (Tools)The Hippocampus: Short-term vs. Long-term MemoryLab: Building a Tool-Using Agent from ScratchIn Chapter 10, we discussed Reinforcement Learning agents that learn by trial and error. That is great for video games, but it is too slow for the real world. You cannot train a stock-trading bot by letting it bankrupt you a million times.Instead, we use LLM-based Agents. We take a pre-trained Large Language Model (which already knows physics, coding, and language) and wrap it in a control loop. We treat the LLM not as a database of facts, but as a Reasoning Engine (CPU).The Architecture of AgencyAn Agent is a system composed of three distinct modules. Think of it as a robot:The Brain (LLM): The controller. It plans, decides, and reflects.The Hands (Tools): The actuators. Python scripts, API calls, Web Browsers.The Notebook (Memory): The state storage. Vector Databases or SQL logs.The goal of Agentic AI is to connect these components so the system can solve multi-step problems without human intervention.The ReAct Loop: The Physics of ThoughtThe fundamental law of Agentic AI is the ReAct pattern (Reason + Act).It transforms the linear flow of a Chatbot into a circular Feedback Loop.Thought: The Agent analyzes the user's request. ("User wants the weather in Tokyo. I should check if I have a weather tool.")Action: The Agent selects a tool and generates the inputs. (weather_tool(city="Tokyo"))Observation: The Agent stops and waits. The code runs. The output ("15°C, Rainy") is fed back into the Agent.Synthesis: The Agent reads the observation and generates the final answer. ("It is currently raining in Tokyo.")This loop allows the Agent to correct its own errors. If the tool fails ("Error: City not found"), the Agent sees that observation and thinks: "Oops, maybe I spelled it Tokoyo. Let me try again."The Action Space: Defining ToolsAn LLM is trapped in a text box. It cannot touch the internet. To give it power, we must define an Interface.We describe tools to the LLM using JSON schemas or Python function signatures.System: "You have a tool called calculate_velocity. It takes distance and time as inputs."When the LLM wants to use this tool, it doesn't run code itself. It outputs a specific text token, like:Action: calculate_velocity(100, 9.8)We (the runtime system) parse that text, execute the actual Python function, and paste the result back into the chat window for the LLM to see.Lab 11.1: The Proto-AgentWe will build a minimal implementation of the ReAct loop in Python. We will simulate the LLM part to show exactly how the logic flows.The Code:Pythonimport re

# 1. The Hands (Tools)
def get_weather(location):
    # In a real agent, this would call an API
    if "Tokyo" in location:
        return "15°C and Rainy"
    elif "New York" in location:
        return "22°C and Sunny"
    else:
        return "Unknown location"

def calculator(expression):
    try:
        return str(eval(expression))
    except:
        return "Error in calculation"

# Registry of available physics tools
tools = {
    "get_weather": get_weather,
    "calculator": calculator
}

# 2. The Brain (Simulated LLM)
# This mimics what GPT-4 would output given the prompt.
def simulated_llm(prompt):
    if "weather in Tokyo" in prompt and "Observation" not in prompt:
        return "Thought: I need to check the weather data.\nAction: get_weather('Tokyo')"
    elif "15°C" in prompt and "Final Answer" not in prompt:
        return "Thought: The weather is rainy. I should advise an umbrella.\nFinal Answer: It is raining in Tokyo, bring an umbrella."
    else:
        return "Final Answer: I am confused."

# 3. The Control Loop (The Agent)
class Agent:
    def __init__(self, llm_func, tools):
        self.llm = llm_func
        self.tools = tools
        self.memory = [] # The Context Window

    def run(self, user_query):
        self.memory.append(f"User: {user_query}")
        print(f"User: {user_query}")
        
        # Limit the loop to prevent infinite recursion (The Halting Problem)
        for step in range(5):
            # Construct the current state of the universe
            context = "\n".join(self.memory)
            
            # 1. Reason (Call LLM)
            response = self.llm(context)
            self.memory.append(response)
            print(f"Agent: {response}")
            
            # Check if finished
            if "Final Answer:" in response:
                return response.split("Final Answer:")[1].strip()
            
            # 2. Act (Parse Action)
            # Regex to find "Action: tool_name('input')"
            action_match = re.search(r"Action: (\w+)\((.*)\)", response)
            if action_match:
                tool_name = action_match.group(1)
                tool_input = action_match.group(2).replace("'", "").replace('"', "")
                
                # Execute Physics
                if tool_name in self.tools:
                    print(f"--- Executing Tool: {tool_name} ---")
                    result = self.tools[tool_name](tool_input)
                    
                    # 3. Observe
                    observation = f"Observation: {result}"
                    self.memory.append(observation)
                    print(f"System: {observation}")
                else:
                    self.memory.append("Observation: Tool not found.")

# Run the Experiment
bot = Agent(simulated_llm, tools)
bot.run("What is the weather in Tokyo?")
Analyzing the Trace:Step 1: The Agent thinks "I need weather" and outputs an Action.Intervention: The code catches that Action, runs the get_weather function, and appends "Observation: 15°C..." to the memory.Step 2: The Agent sees the observation. It now has the information required to answer. It outputs "Final Answer."Memory Systems: The HippocampusOur self.memory list above is Short-Term Memory. It lives only as long as the Python script runs.For a true autonomous agent, we need Long-Term Memory.We use Vector Databases (like Pinecone or ChromaDB).When the Agent learns a fact ("The user loves Sushi"), we embed it (turn it into numbers) and store it in the database.Weeks later, when the user asks for a restaurant recommendation, the Agent queries the database for "User preferences."The relevant memories are retrieved and injected into the Short-Term context.THEORY LAB: Retrieval Augmented Generation (RAG).This process is analogous to a computer accessing its Hard Drive (Vector DB) to load data into RAM (Context Window) for the CPU (LLM) to process.Orchestration Frameworks: LangChain & LangGraphWriting raw loops (like our Lab above) is messy. In production, we use frameworks.LangChain: A library of standard "pipes" and "chains" to connect LLMs to tools.LangGraph: A newer approach that models agents as State Machines (Nodes and Edges). This allows for complex workflows:If tool fails $\rightarrow$ Go to Node B (Error Handler).If tool succeeds $\rightarrow$ Go to Node C (Summarizer).WARNING: Infinite Loops.A common failure mode in Agentic AI is the loop where the Agent tries to fix an error, causes the same error, and tries again forever. You must always implement a max_iterations counter (Entropy Death) to kill the process if it gets stuck.Coming Up NextOne agent is powerful. But what about a team of agents?In Chapter 12, we will explore Swarm Dynamics. We will build Multi-Agent Systems where a "Researcher" agent hands off work to a "Writer" agent, overseen by a "Manager" agent.