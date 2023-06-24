import tkinter as tk
from tkinter import ttk
from main import SpamFilter # assuming main.py and app.py are in the same directory

class EmailClassifierApp:
    def __init__(self, spam_filter):
        self.spam_filter = spam_filter
        self.window = tk.Tk()
        self.window.title("Email Spam Classifier")

        self.email_text_label = ttk.Label(self.window, text="Enter Email Text:")
        self.email_text_label.pack()

        self.email_text_field = ttk.Entry(self.window, width=50)
        self.email_text_field.pack()

        self.classify_button = ttk.Button(self.window, text="Classify", command=self.classify_email)
        self.classify_button.pack()

        self.result_label = ttk.Label(self.window, text="")
        self.result_label.pack()

    def classify_email(self):
        email_text = self.email_text_field.get()
        result = self.spam_filter.predict(email_text)
        self.result_label.config(text=f"This email is likely: {result}")

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    spam_filter = SpamFilter('/Users/hyunerickang/Desktop/ML Email Spam Filter/spam_ham_dataset.csv')
    spam_filter.train_model()

    app = EmailClassifierApp(spam_filter)
    app.run()
