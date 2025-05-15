import PyPDF2
from langchain import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import override

from BaseAgent import BaseAgent


class CodingAgent(BaseAgent):
    role = "Coding Agent"
    basic_prompt_template = """
    You are a proficient CodingAgent.
    Based on the instructions provided generate a well-structured and fully commented code module.
    The module must be designed to be integrated directly into a self-contained HTML file that includes inline CSS and JavaScript.
    Write all the code as per the instructions provided. Incase there is no code given then start writing the whole code module from scratch.
    RETURN ONLY THE CODE AND NO pre and post text other than code
    """

    coding_instructions = None
    previous_code_module = None

    def __init__(self, coding_instructions,previous_code_module="No code till now !!"):
        self.coding_instructions = coding_instructions.strip() if coding_instructions else ""
        prompt = self.basic_prompt_template.format(coding_instructions=self.coding_instructions)
        super(CodingAgent, self).__init__(self.role, prompt, context=None)
        self.previous_code_module = previous_code_module

    @override
    def get_output(self):
        if not self.llm:
            raise ValueError("LLM is not set.")

        # Use the enhanced prompt if available, else the basic one
        base_prompt = self.enhanced_prompt if self.enhanced_prompt else self.basic_prompt

        final_prompt_template = (
            "You are an expert in {role}.\n\n"
            "{context}\n\n"
            "Here is the task given to you: \n"
            "{base_prompt}\n"
            "Coding Instructions:\n"
            "{coding_instructions}\n"
            "Code till Now\n"
            "{previous_code_module}\n"
        )

        prompt = PromptTemplate(
            input_variables=["role", "context", "base_prompt", "coding_instructions","previous_code_module"],
            template=final_prompt_template
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.invoke({
            "role": self.role,
            "context": self.context,
            "base_prompt": base_prompt,
            "coding_instructions": self.coding_instructions,
            "previous_code_module": self.previous_code_module
        })['text']


if __name__ == "__main__":
    print("This is a Coding Agent module. It is not meant to be run directly.")
    coding_instructions = """
    kay, here is the comprehensive Implementation Plan for the Process Lifecycle and Context Switching Simulator, adhering strictly to the provided requirements and the single-file constraint.

**Implementation Plan: Process Lifecycle & Context Switching Simulator**

**1. High-Level Design**

*   **Overall Architecture:**
    *   The simulator will be delivered as a single HTML file (`simulator.html`).
    *   All presentation logic (styling) will be embedded within `<style>` tags in the `<head>` section of the HTML file. Standard CSS will be used for layout and visual state representation (e.g., different background colors for process states).
    *   All behavioral logic (simulation engine, UI manipulation, state management) will be embedded within `<script>` tags, placed just before the closing `</body>` tag to ensure the DOM is loaded before the script executes. No external JavaScript libraries will be used unless they are trivially embeddable and essential (standard vanilla JavaScript is preferred).
    *   The HTML structure will use semantic `div` elements for layout:
        *   A main container `div` (`id="simulator-container"`).
        *   A `div` for user controls (`id="controls"`).
        *   A `div` to display the list of processes (`id="process-list-display"`).
        *   A `div` to show details of a selected/relevant Process Control Block (`id="pcb-details-display"`).
        *   A `div` to display the event log (`id="event-log-display"`).

*   **Core Data Structures:**
    *   **Process Control Block (PCB):** A JavaScript object representing a single process. Key properties will include:
        *   `processId`: (String/Number) Unique identifier (e.g., "P1", "P2" or timestamp-based).
        *   `state`: (String) Current state (e.g., 'New', 'Ready', 'Running', 'Waiting', 'Terminated').
        *   `programCounter`: (Number) Simulated instruction pointer.
        *   `registers`: (Object) A simple object simulating CPU registers (e.g., `{ R1: 0, R2: 0, ... }` or just placeholder values).
        *   `memoryPointers`: (Object) Simulated memory allocation info (e.g., `{ base: 1024, limit: 2048 }` or simplified placeholders).
        *   *Other relevant data:* Could include priority, creation time, etc., if needed later, but stick to basics initially based on requirements.
    *   **Process Collection:** A JavaScript array holding all active PCB objects.
        *   `let processList = [];`
    *   **Queues (Conceptual/Actual):** While not strictly required to be separate arrays, managing 'Ready' and 'Waiting' states implies a need to easily find processes in these states. We might maintain separate arrays or filter `processList` as needed.
        *   `let readyQueue = []; // Array of processIds`
        *   `let waitingQueue = []; // Array of processIds`

*   **State Management:**
    *   The global JavaScript scope (or a dedicated simulation state object) will hold the `processList`, `readyQueue`, `waitingQueue`.
    *   A variable will track the currently 'Running' process ID:
        *   `let currentRunningProcessId = null;`
    *   The state of each process is stored directly within its PCB object (`process.state`).
    *   The overall simulation state is implicitly defined by the contents of `processList` and `currentRunningProcessId`. UI updates will read directly from these structures.

**2. Key Modules/Components (Conceptual)**

These will be logical groupings of functions/objects within the single `<script>` tag, delineated by comments for clarity.

*   **`UIManager`:**
    *   **Responsibilities:** Handles all interactions with the DOM. Renders the process list, updates process visuals based on state, displays PCB details for a selected/running process, updates the event log display, attaches event listeners to control buttons.
    *   **Interactions:** Called by `SimulationEngine` and `ProcessManager` to update the view after state changes. Calls `ProcessManager` or `SimulationEngine` functions in response to user button clicks.
*   **`ProcessManager`:**
    *   **Responsibilities:** Manages the `processList` array. Handles process creation (generating IDs, initializing PCBs), finding processes by ID, potentially removing terminated processes.
    *   **Interactions:** Called by `UIManager` (on user 'Create Process' action). Called by `SimulationEngine` to get/update process data.
*   **`SimulationEngine`:**
    *   **Responsibilities:** Contains the core simulation logic. Implements state transition functions (`moveToReady`, `moveToRunning`, `moveToWaiting`), the `contextSwitch` logic, and the `stepSimulation` logic. Orchestrates changes to process states and triggers UI updates via `UIManager`. Manages the `readyQueue`, `waitingQueue`, and `currentRunningProcessId`.
    *   **Interactions:** Called by `UIManager` (on user 'Context Switch' or 'Step' actions). Calls `ProcessManager` to get process objects. Calls `UIManager` to refresh the display after actions. Calls `EventLogger`.
*   **`EventLogger`:**
    *   **Responsibilities:** Provides a function to record significant events (process creation, state changes, context switches, user actions, validation results). Stores log messages in an array.
    *   **Interactions:** Called by `UIManager`, `ProcessManager`, and `SimulationEngine` to log events. Read by `UIManager` to display the log.

**3. Specific Coding Instructions & Logic**

*   **Process Lifecycle Simulation:**
    *   `createProcess()`:
        *   Generate a unique `processId`.
        *   Create a new PCB object with default values (e.g., `state: 'New'`, `programCounter: 0`, initial register values).
        *   Add the new PCB to `processList`.
        *   Transition the process immediately to 'Ready' state: `moveToReady(newProcessId)`.
        *   Log the creation event using `EventLogger.log()`.
        *   Trigger `UIManager.renderProcessList()` and potentially `UIManager.renderPCBDetails()` if applicable.
    *   `moveToReady(processId)`:
        *   Find PCB by `processId` in `processList`.
        *   If found and state is appropriate (e.g., 'New', 'Running', 'Waiting'):
            *   Update `pcb.state = 'Ready'`.
            *   Add `processId` to `readyQueue` (ensure no duplicates).
            *   Remove `processId` from `waitingQueue` if present.
            *   If `processId` was `currentRunningProcessId`, set `currentRunningProcessId = null`.
            *   Log state change.
            *   Trigger UI update for the specific process.
    *   `moveToRunning(processId)`:
        *   Find PCB by `processId`.
        *   Ensure no other process is 'Running' (`currentRunningProcessId === null`).
        *   If found and state is 'Ready':
            *   Update `pcb.state = 'Running'`.
            *   Set `currentRunningProcessId = processId`.
            *   Remove `processId` from `readyQueue`.
            *   Log state change.
            *   Trigger UI update (process list, PCB details).
    *   `moveToWaiting(processId)`:
        *   Find PCB by `processId`.
        *   If found and state is 'Running':
            *   Update `pcb.state = 'Waiting'`.
            *   Add `processId` to `waitingQueue`.
            *   If `processId` was `currentRunningProcessId`, set `currentRunningProcessId = null`.
            *   Log state change.
            *   Trigger UI update.
    *   `terminateProcess(processId)`: (Optional, but good practice)
        *   Find PCB by `processId`.
        *   Update `pcb.state = 'Terminated'`.
        *   Remove from `readyQueue`, `waitingQueue`.
        *   If `currentRunningProcessId === processId`, set `currentRunningProcessId = null`.
        *   Potentially remove from `processList` after a delay or keep as 'Terminated' for display.
        *   Log termination.
        *   Trigger UI update.

*   **Context Switching:**
    *   `contextSwitch(outgoingProcessId, incomingProcessId)`:
        *   **Save Outgoing Context:**
            *   Find `outgoingPCB` using `outgoingProcessId`.
            *   *Simulate Save:* Copy relevant "CPU state" (e.g., current `programCounter`, `registers` - these might be conceptually held in `SimulationEngine` or just directly updated in the PCB) *into* the `outgoingPCB` object's properties (`outgoingPCB.programCounter = currentPC; outgoingPCB.registers = currentRegisters;`).
            *   Log: "Saved context for Process " + `outgoingProcessId`.
        *   **Load Incoming Context:**
            *   Find `incomingPCB` using `incomingProcessId`.
            *   *Simulate Load:* Copy relevant properties *from* the `incomingPCB` object *to* the conceptual "CPU state" (`currentPC = incomingPCB.programCounter; currentRegisters = { ...incomingPCB.registers };`).
            *   Log: "Loaded context for Process " + `incomingProcessId`.
        *   **Update States:**
            *   Call appropriate state transition functions (e.g., `moveToWaiting(outgoingProcessId)` or `moveToReady(outgoingProcessId)`, and `moveToRunning(incomingProcessId)`).
        *   **Validation:** Perform conceptual validation: Check if `outgoingPCB.programCounter` was updated *before* `currentPC` was overwritten by the incoming process's data. Log validation success/failure.
        *   Log: "Context Switch complete: " + `outgoingProcessId` + " -> " + `incomingProcessId`.
        *   Trigger `UIManager` updates.

*   **Data Structure Manipulation:**
    *   All modifications to PCBs (state, pc, registers) happen directly on the objects within the `processList` array.
    *   Functions like `moveToReady`, `contextSwitch` will directly access `processList.find(p => p.processId === id)` and modify the returned object's properties.
    *   `readyQueue` and `waitingQueue` arrays will be updated using standard array methods (`push`, `shift`, `filter`, `splice`).

*   **UI Rendering & Updates:**
    *   `UIManager.renderProcessList()`:
        *   Get the `process-list-display` div.
        *   Clear its current content (`innerHTML = ''`).
        *   Iterate through `processList`.
        *   For each `pcb`:
            *   Create a `div` element (`document.createElement('div')`).
            *   Assign a unique ID: `element.id = "process-" + pcb.processId`.
            *   Add CSS classes: `element.classList.add('process', 'state-' + pcb.state.toLowerCase())`.
            *   Set content: `element.textContent = \`Process ${pcb.processId} (${pcb.state})\``.
            *   (Optional: Add click listener to show PCB details).
            *   Append the element to the `process-list-display` div.
    *   `UIManager.renderPCBDetails(processId)`:
        *   Get the `pcb-details-display` div.
        *   Find the `pcb` using `processId`.
        *   If found, format its properties (ID, State, PC, Registers, Memory) into HTML (e.g., using a `<pre>` tag or definition list `<dl>`).
        *   Set the `innerHTML` of the `pcb-details-display` div.
        *   If `processId` is null or not found, clear the display.
    *   `UIManager.renderLog()`:
        *   Get the `event-log-display` div.
        *   Get the log messages from `EventLogger`.
        *   Format messages (e.g., each on a new line).
        *   Set the `innerHTML` or `textContent` of the log display div. Ensure it scrolls (e.g., using CSS `overflow-y: auto; height: ...;`).

*   **User Controls:**
    *   Add HTML buttons with IDs: `<button id="create-process-btn">Create Process</button>`, `<button id="context-switch-btn">Context Switch Running -> Waiting</button>`, `<button id="step-simulation-btn">Step Simulation</button>`.
    *   In JavaScript, add event listeners:
        *   `document.getElementById('create-process-btn').addEventListener('click', () => { SimulationEngine.requestCreateProcess(); });` (Wrapper function might be needed in Engine/Manager).
        *   `document.getElementById('context-switch-btn').addEventListener('click', () => { SimulationEngine.requestContextSwitchRunningToWaiting(); });`
        *   `document.getElementById('step-simulation-btn').addEventListener('click', () => { SimulationEngine.step(); });`
    *   `SimulationEngine.requestCreateProcess()`: Calls `ProcessManager.createProcess()`, logs action, triggers UI update.
    *   `SimulationEngine.requestContextSwitchRunningToWaiting()`:
        *   Check if `currentRunningProcessId` is not null.
        *   If yes:
            *   Log user action.
            *   Identify the `outgoingProcessId = currentRunningProcessId`.
            *   Select the next process from `readyQueue` (e.g., `incomingProcessId = readyQueue[0]`). Handle empty queue case.
            *   If an `incomingProcessId` is found:
                *   Call `contextSwitch(outgoingProcessId, incomingProcessId)`. The `contextSwitch` function itself should handle moving the outgoing process to 'Waiting' (as per button intent) and incoming to 'Running'.
            *   If no `incomingProcessId` (ready queue empty), just move the running process to waiting: `moveToWaiting(outgoingProcessId)`.
        *   If no process is running, log an info message.
    *   `SimulationEngine.step()`:
        *   Check if `currentRunningProcessId` is not null.
        *   If yes:
            *   Find the running PCB.
            *   *Simulate work:* Increment `pcb.programCounter`.
            *   Log the step event.
            *   Trigger `UIManager.renderPCBDetails(currentRunningProcessId)` to show the updated PC.

*   **Logging:**
    *   `EventLogger`: Maintain an array `let logMessages = [];`.
    *   `log(message)`:
        *   Add timestamp prefix (optional but helpful): `const timestamp = new Date().toLocaleTimeString();`
        *   `logMessages.push(\`${timestamp}: ${message}\`);`
        *   Trigger `UIManager.renderLog()`.

*   **Validation Logic:**
    *   Implement checks within `contextSwitch`:
        *   Before loading incoming context, conceptually verify that the outgoing PCB object's relevant fields (e.g., `programCounter`) have been updated by the "save" step. `if (outgoingPCB.programCounter === valueJustSaved) { log("Validation: Outgoing context save successful."); } else { log("Validation ERROR: Outgoing context save failed."); }`
        *   After loading incoming context, check if the conceptual "CPU state" (`currentPC`) matches the value from the incoming PCB. `if (currentPC === incomingPCB.programCounter) { log("Validation: Incoming context load successful."); } else { log("Validation ERROR: Incoming context load failed."); }`
    *   Implement checks in state transitions: Ensure a process is in a valid state before transitioning (e.g., cannot move from 'New' directly to 'Waiting'). Log validation failures.

**4. Implementation Strategy & Considerations**

*   **Single-File Structure:**
    *   HTML: Standard structure (`<!DOCTYPE html>`, `<html>`, `<head>`, `<body>`).
    *   CSS: Place all styles within `<style>...</style>` in the `<head>`. Use classes extensively for styling states (`.process.state-running`, `.process.state-ready`, etc.).
    *   JavaScript: Place all code within `<script>...</script>` just before `</body>`. Use comments (`// --- Module: UIManager ---`, `// --- Process Creation Logic ---`) to logically separate conceptual modules (`UIManager`, `ProcessManager`, `SimulationEngine`, `EventLogger`) and functions. Use IIFE (Immediately Invoked Function Expression) to avoid polluting the global scope if desired `(function() { /* all sim code */ })();`.
*   **DOM Manipulation:**
    *   Use `document.getElementById` for accessing main containers and controls.
    *   For the process list, initially, clearing and re-rendering the entire list (`process-list-display.innerHTML = ''` followed by a loop) is simplest. If performance degrades with many processes (unlikely for a basic simulator), transition to finding and updating individual process `div`s using their unique IDs (`document.getElementById('process-' + pcb.processId)`).
    *   Update PCB details and log similarly by setting `innerHTML` of their respective containers.
*   **Event Handling:**
    *   Attach event listeners directly to the control buttons using `addEventListener` after the DOM is loaded.
    *   If processes in the list need to be clickable (e.g., to select for detailed view), use event delegation: attach a single listener to the `process-list-display` container and check `event.target` to identify which process `div` was clicked.

This plan provides a detailed blueprint for developing the simulator within the specified constraints, focusing on vanilla JavaScript, embedded CSS, and clear logical separation within a single HTML file.

    """
    coding_agent = CodingAgent(coding_instructions)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro-exp-03-25",
        temperature=0.1,
        max_tokens=100000
    )
    coding_agent.set_llm(llm)
    coding_agent.set_prompt_enhancer_llm(llm)
    print("_" * 50)
    print(coding_agent.enhance_prompt())
    print("_" * 50)
    print(coding_agent.get_output())
