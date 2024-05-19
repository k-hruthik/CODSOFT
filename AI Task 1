data = {
 "hi": "Hi there! I'm a friendly chatbot here to assist you.",
    "hello": "Hello! How can I help you today?",
    "what is your name": "I'm just a chatbot, so I don't have a name, but you can call me ChatBot.",
    "where are you from": "I'm from the digital world, always ready to chat!",
    "how are you": "I'm just a chatbot, but I'm here to assist you.",
    "do you have any hobbies or interests?": "I'm always busy helping users, so my hobby is chatting with people like you!",
    "what did you eat today?": "I don't eat, but I can help you find delicious recipes and food-related information.",
    "what's your favorite color?": "I'm a chatbot, so I don't have personal preferences for colors.",
    "do you enjoy listening to music?": "I can't listen to music, but I'm here to chat about it!",
    "bye": "Bye! Take care and have a great day!",
    "what can you do?": "I can chat with you, provide information, answer questions, and assist you with various tasks.",
    "what is the weather like today?": "I'm not able to check the weather, but you can ask a weather service or check your weather app.",
    "tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
    "what is your favorite movie?": "I don't watch movies, but I can help you find information about any movie you're interested in.",
    "can you help me with my homework?": "Sure! I can try to assist you with your homework. What subject are you working on?",
    "how old are you?": "I don't have an age. I'm just a program created to assist you.",
    "what is your purpose?": "My purpose is to assist users by providing information and answering questions.",
    "do you know any quotes?": "Here's one for you: 'The only way to do great work is to love what you do.' - Steve Jobs",
    "can you tell me a story?": "Once upon a time in a digital world, there was a friendly chatbot who loved to help users. It spent its days chatting and providing information, always ready to assist those in need. And it lived happily ever after!",
    "how do you work?": "I work by processing your input and responding based on pre-defined patterns and responses.",
    "who created you?": "I was created by Hruthik.",
    "what's your favorite book?": "I don't read books, but I can help you find information about any book you're interested in.",
    "can you speak other languages?": "I can understand and respond in several languages. Try asking me something in another language!",
    "what's the time now?": "I can't tell the time, but you can check the time on your device.",
    "how can I learn programming?": "You can start learning programming with online tutorials, courses, and books. Websites like Codecademy, Coursera, and Khan Academy are great places to start.",
    "what's the capital of France?": "The capital of France is Paris.",
    "do you like sports?": "I don't play or watch sports, but I can chat about them with you!",
    "who is your favorite author?": "I don't have personal preferences, but I can help you find information about any author.",
    "can you recommend a movie?": "Sure! What genre are you interested in?",
    "what's your favorite food?": "I don't eat, but I can help you find information about various cuisines and recipes.",
    "how do I stay healthy?": "Staying healthy involves eating a balanced diet, exercising regularly, getting enough sleep, and managing stress.",
    "what's the best way to learn a new language?": "The best way to learn a new language is through regular practice, using language learning apps, taking classes, and immersing yourself in the language by reading, listening, and speaking it as much as possible.",
    "can you play games?": "I can't play games, but I can recommend some fun games for you to try!",
    "what's the meaning of life?": "The meaning of life is a philosophical question that has been debated for centuries. Different people and cultures have different beliefs about it.",
    "can you solve math problems?": "Sure! Ask me a math problem, and I'll do my best to help.",
    "what's the latest news?": "I can't browse the internet for the latest news, but you can check a news website or app for the most recent updates.",
    "do you believe in aliens?": "As a chatbot, I don't have personal beliefs, but the existence of aliens is a topic of scientific investigation and speculation.",
    "what's your favorite animal?": "I don't have personal preferences, but I can help you find information about any animal.",
    "what's your favorite subject?": "I don't have personal preferences for subjects, but I can help you with any subject you're interested in.",
    "can you tell me about yourself?": "I'm a chatbot created by Hruthik to assist users by providing information and answering questions.",
    "what is the meaning of love?": "Love is a complex and deeply personal emotion that can take many forms. It's often described as a deep affection, attachment, or strong feeling of care towards someone or something.",
    "how do I improve my productivity?": "Improving productivity involves setting clear goals, managing your time effectively, minimizing distractions, and staying organized.",
    "what's the difference between AI and machine learning?": "Artificial Intelligence (AI) is a broad field of computer science focused on creating systems that can perform tasks that would typically require human intelligence. Machine Learning is a subset of AI that involves training algorithms to learn patterns and make predictions from data.",
    "what's your favorite place?": "I don't have personal preferences for places, but I can help you find information about any location you're interested in.",
    "can you give me some writing tips?": "Sure! Some writing tips include outlining your ideas before you start, revising and editing your work, and reading widely to improve your vocabulary and style.",
    "what's the difference between a robot and a chatbot?": "A robot is a physical machine that can perform tasks autonomously or under remote control. A chatbot is a software program designed to simulate conversation with human users over the internet.",
    "what's the best way to start a conversation?": "Starting a conversation can be as simple as asking a question, making a comment about the environment or situation, or sharing something about yourself.",
    "can you recommend a good podcast?": "Sure! What topics are you interested in?",
    "what's the difference between empathy and sympathy?": "Empathy is the ability to understand and share the feelings of others, while sympathy is."
}

def get_response(user_input):
    for pattern, response in data.items():
        if pattern in user_input.lower():
            return response
    return "I'm sorry, I didn't understand that. Can you please rephrase your sentence?"

print("Chatbot: Hi! I'm a simple chatbot, I'm here to assist you!")
while True:
    user_input = input("Me: ")
    if user_input.lower() == 'bye':
        print("Chatbot: Goodbye! Have a great day!")
        break
    response = get_response(user_input)
    print("Chatbot:", response)
