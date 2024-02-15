import tkinter as tk
from tkinter import ttk
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialize NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Initialize PRAW with your Reddit API credentials
reddit = praw.Reddit(client_id='XXXXXXXXXXXX',
                     client_secret='XXXXXXXXXXXX',
                     user_agent='XXXXXXXXXXX')

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize NLTK's Snowball Stemmer and stopwords
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

# Lists to store sentiment values
positive_sentiments = []
negative_sentiments = []
neutral_sentiments = []

# Lists to store top comments
top_positive_comment = ""
top_negative_comment = ""
top_neutral_comment = ""

# Create a function to perform sentiment analysis and visualization
def analyze_and_visualize():
    subreddit_name = subreddit_entry.get().strip()  # Remove leading and trailing spaces
    num_posts = int(posts_entry.get())

    for submission in reddit.subreddit(subreddit_name).new(limit=num_posts):
        total_sentiment = 0
        total_comments = 0
        positive_comments = []
        negative_comments = []
        neutral_comments = []

        for comment in submission.comments:
            if isinstance(comment, praw.models.Comment):
                comment_text = comment.body.lower()
                comment_text = comment_text.translate(str.maketrans('', '', string.punctuation))
                words = nltk.word_tokenize(comment_text)
                words = [stemmer.stem(word) for word in words if word not in stop_words]
                cleaned_comment = ' '.join(words)

                sentiment = analyzer.polarity_scores(cleaned_comment)
                total_sentiment += sentiment['compound']
                total_comments += 1

                if sentiment['compound'] >= 0.05:
                    positive_comments.append(comment.body)
                elif sentiment['compound'] <= -0.05:
                    negative_comments.append(comment.body)
                else:
                    neutral_comments.append(comment.body)

        if total_comments > 0:
            average_sentiment = total_sentiment / total_comments
            if average_sentiment >= 0.05:
                positive_sentiments.append(average_sentiment)
                top_positive_comment = get_top_comment(positive_comments, sentiment['compound'])
            elif average_sentiment <= -0.05:
                negative_sentiments.append(average_sentiment)
                top_negative_comment = get_top_comment(negative_comments, sentiment['compound'])
            else:
                neutral_sentiments.append(average_sentiment)
                top_neutral_comment = get_top_comment(neutral_comments, sentiment['compound'])

    # Calculate percentages
    total_count = len(positive_sentiments) + len(negative_sentiments) + len(neutral_sentiments)
    positive_percentage = (len(positive_sentiments) / total_count) * 100
    negative_percentage = (len(negative_sentiments) / total_count) * 100
    neutral_percentage = (len(neutral_sentiments) / total_count) * 100

    # Clear existing data
    clear_top_comments()

    # Create and display the bar chart
    create_bar_chart(positive_percentage, negative_percentage, neutral_percentage)

    # Create and display the pie chart
    create_pie_chart(positive_percentage, negative_percentage, neutral_percentage)

    # Display top comments with styling
    top_comments_label.config(text="Top Comments", font=("Arial", 14, "bold"))
    add_top_comment(top_positive_comment, 'green', top_positive_comment_label)
    add_top_comment(top_negative_comment, 'red', top_negative_comment_label)
    add_top_comment(top_neutral_comment, 'blue', top_neutral_comment_label)

# Create a function to create and display a bar chart
def create_bar_chart(positive_percentage, negative_percentage, neutral_percentage):
    bar_chart_frame = ttk.Frame(graph_section)
    bar_chart_frame.grid(row=0, column=0, padx=10, pady=10)

    labels = ['Positive', 'Negative', 'Neutral']
    percentages = [positive_percentage, negative_percentage, neutral_percentage]

    plt.figure(figsize=(5, 4))
    plt.bar(labels, percentages)
    plt.ylabel('Percentage')
    plt.title('Sentiment Analysis Distribution (Bar Chart)')
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(plt.gcf(), master=bar_chart_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Create a function to create and display a pie chart
def create_pie_chart(positive_percentage, negative_percentage, neutral_percentage):
    pie_chart_frame = ttk.Frame(graph_section)
    pie_chart_frame.grid(row=0, column=1, padx=10, pady=10)

    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [positive_percentage, negative_percentage, neutral_percentage]
    colors = ['#66b3ff', '#ff9999', '#99ff99']

    plt.figure(figsize=(5, 4))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Sentiment Analysis Distribution (Pie Chart)')
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(plt.gcf(), master=pie_chart_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Create a function to get the top comment for each sentiment
def get_top_comment(comments, sentiment):
    if not comments:
        return "No comments in this category."

    # Find the comment with the highest positive or negative sentiment
    top_comment = max(comments, key=lambda x: analyzer.polarity_scores(x)["compound"])

    return top_comment

# Create a function to add a top comment
def add_top_comment(comment, color, label):
    label.config(text=comment, font=("Arial", 12), fg=color, justify="left")

# Create a function to clear existing top comments
def clear_top_comments():
    top_positive_comment_label.config(text="", font=("Arial", 12), fg="green")
    top_negative_comment_label.config(text="", font=("Arial", 12), fg="red")
    top_neutral_comment_label.config(text="", font=("Arial", 12), fg="blue")

# Create a Tkinter window
window = tk.Tk()
window.title("Sentiment Analysis App")
window.geometry("1920x1080")  # Set the initial window size

# Create a main frame to hold all sections
main_frame = ttk.Frame(window)
main_frame.pack(fill=tk.BOTH, expand=True)

# Create a canvas to contain the sections and add scrollbars
canvas = tk.Canvas(main_frame)
canvas.pack(side="left", fill="both", expand=True)

scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
scrollbar.pack(side="right", fill="y")

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Create frames for each section (input, graph, top comments)
input_section = ttk.Frame(canvas)
graph_section = ttk.Frame(canvas)
top_comments_section = ttk.Frame(canvas)

canvas.create_window((0, 0), window=input_section, anchor="nw")
canvas.create_window((0, 275), window=graph_section, anchor="nw")
canvas.create_window((0, 125), window=top_comments_section, anchor="nw")

# Create input fields for subreddit name and number of posts
subreddit_label = tk.Label(input_section, text="Subreddit Name:", font=("Arial", 12))
subreddit_label.grid(row=0, column=0, padx=10, pady=10)
subreddit_entry = tk.Entry(input_section)
subreddit_entry.grid(row=0, column=1, padx=10, pady=10)

posts_label = tk.Label(input_section, text="Number of Posts to Analyze:", font=("Arial", 12))
posts_label.grid(row=1, column=0, padx=10, pady=10)
posts_entry = tk.Entry(input_section)
posts_entry.grid(row=1, column=1, padx=10, pady=10)

# Create a button to trigger the analysis
analyze_button = tk.Button(input_section, text="Analyze and Visualize", command=analyze_and_visualize, font=("Arial", 12, "bold"))
analyze_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Label to display top comments
top_comments_label = tk.Label(top_comments_section, text="Top Comments", font=("Arial", 14, "bold"))
top_comments_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="w")

top_positive_comment_label = tk.Label(top_comments_section, text="Top Positive", font=("Arial", 12), fg="green", justify="left")
top_positive_comment_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

top_negative_comment_label = tk.Label(top_comments_section, text="Top Negative", font=("Arial", 12), fg="red", justify="left")
top_negative_comment_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")

top_neutral_comment_label = tk.Label(top_comments_section, text="Top Neutral", font=("Arial", 12), fg="blue", justify="left")
top_neutral_comment_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")

# Start the Tkinter main loop
window.mainloop()
