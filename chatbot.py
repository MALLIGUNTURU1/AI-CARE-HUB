import tkinter as tk
import google.generativeai as genai

# Configure the API key
genai.configure(api_key="AIzaSyBL-dM0DuwFQpo1S5eu4eWSSGR_tLs0bX0")

# Initialize the generative model
model = genai.GenerativeModel('gemini-pro')

def send_message(event=None):
    user_input = user_entry.get()
    user_entry.delete(0, tk.END)

    try:
        # Generate response
        response = model.generate_content(user_input)
        chat_message = response.text

        # Update conversation display
        conversation_text.insert(tk.END, f"User: {user_input}\nBot: {chat_message}\n\n")

    except Exception as e:
        conversation_text.insert(tk.END, f"Error: {str(e)}\n\n")

# Create main window
root = tk.Tk()
root.title("Chatbot")
root.geometry("800x600")
root.configure(bg="#FFFFFF")

# Conversation display
conversation_text = tk.Text(root, wrap=tk.WORD, bg="#F0F0F0", bd=0, font=("Arial", 12), padx=10, pady=10)
conversation_text.pack(expand=True, fill="both")

# User input entry
user_entry = tk.Entry(root, width=50, bd=0, font=("Arial", 12), bg="#FFFFFF")
user_entry.pack(pady=10)
user_entry.bind("<Return>", send_message)

# Send button
send_button = tk.Button(root, text="Send", command=send_message, bg="#4CAF50", fg="#FFFFFF",
                        width=30, bd=0, font=("Arial", 12, "bold"))
send_button.pack(pady=10)

root.mainloop()