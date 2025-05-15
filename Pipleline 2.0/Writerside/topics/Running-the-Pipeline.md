# Running the Pipeline

This topic covers the procedures to start the application and see how the pipeline functions interactively.

## Start the Streamlit Application

To run the interactive user interface:

1. **Launch the Application**  
   Open a terminal in your project directory and execute:

   ```bash
   streamlit run ui.py
   ```
2. **Opening in Web Browser**  
   After running the command, a new tab will open in your default web browser. If it doesn't, you can manually navigate
   to the website shown in the console/terminal.

## Various Elements

1. **Requirements Generation**
    - You can either drag and drop the PDF file to it or use the Button to search for the file using the System File Explorer.
    - Click "Generate Requirements" to extract and display requirements from the PDF.
    - The output is shown in a text area for further review.

2. **Human Review for Requirements**
    - Input your review or feedback in the provided text field.
    - Submit your review to update and refine the requirements.

3. **Implementation Generation**
    - Upon reviewing the requirements, click the "Generate Implementation" button.
    - The application then produces implementation details based on the reviewed requirements.

4. **Iterative Code Generation and Review**
    - Enter your code review feedback, then click "Generate/Refine Code" for iterative improvements.
    - The code output is displayed and highlighted.
    - A link is provided that opens a live preview (served locally at port 8000).

5. **Documentation Generation**
    - After confirming the code, generate documentation to explain and complement the code.

## Live Code Preview

- **Automatic Server Startup:**  
  When code is generated, the script automatically starts a local HTTP server if not already running.
- **Opening in Browser:**  
  The generated URL (e.g., `http://localhost:8000/code.html`) is launched in your default browser, offering immediate
  visual feedback.
- **Manual Navigation:**  
  If the browser does not open automatically, scroll down and copy and paste the URL into your browser manually.

<tip>
Please do not spam the buttons.This might just cause a lot of requests together and you might be rate-limited by Google.
Normally the application should be able to handle the requests and you should not have to click the buttons multiple times.
</tip>