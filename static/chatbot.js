document.addEventListener('DOMContentLoaded', function() {
    const chatIcon = document.getElementById('chatIcon');
    const chatContainer = document.getElementById('chatContainer');
    const minimizeChat = document.getElementById('minimizeChat');
    const userInput = document.getElementById('userInput');
    const sendMessage = document.getElementById('sendMessage');
    const chatMessages = document.getElementById('chatMessages');
    const typingIndicator = document.querySelector('.typing-indicator');

    let isTyping = false;

    // Toggle chat window
    chatIcon.addEventListener('click', () => {
        chatContainer.classList.toggle('active');
        document.querySelector('.notification-badge').style.display = 'none';
    });

    // Minimize chat window
    minimizeChat.addEventListener('click', () => {
        chatContainer.classList.remove('active');
    });

    // Send message on button click
    sendMessage.addEventListener('click', () => {
        sendUserMessage();
    });

    // Send message on Enter key
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendUserMessage();
        }
    });

    function sendUserMessage() {
        const message = userInput.value.trim();
        if (message === '') return;

        // Add user message to chat
        addMessage(message, 'user');
        userInput.value = '';

        // Show typing indicator
        showTypingIndicator();

        // Send message to server
        fetch('/chatbot/message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            hideTypingIndicator();
            if (data.error) {
                addMessage('Sorry, I encountered an error. Please try again.', 'bot');
            } else {
                addMessage(data.response, 'bot');
            }
        })
        .catch(error => {
            hideTypingIndicator();
            addMessage('Sorry, I encountered an error. Please try again.', 'bot');
            console.error('Error:', error);
        });
    }

    // Load chat history from localStorage
    function loadChatHistory() {
        const history = localStorage.getItem('chatHistory');
        if (history) {
            const messages = JSON.parse(history);
            chatMessages.innerHTML = ''; // Clear default message
            messages.forEach(msg => {
                addMessage(msg.text, msg.sender, msg.time, false);
            });
        }
    }

    // Save message to history
    function saveToHistory(text, sender, time) {
        const history = localStorage.getItem('chatHistory');
        const messages = history ? JSON.parse(history) : [];
        messages.push({ text, sender, time });
        localStorage.setItem('chatHistory', JSON.stringify(messages));
    }

    // Update addMessage function
    function addMessage(text, sender, time = null, save = true) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        const icon = document.createElement('i');
        icon.className = sender === 'bot' ? 'fas fa-robot bot-icon' : 'fas fa-user user-icon';

        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        messageText.innerHTML = text;

        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = time || getFormattedTime();

        messageContent.appendChild(sender === 'bot' ? icon : messageText);
        messageContent.appendChild(sender === 'bot' ? messageText : icon);
        messageDiv.appendChild(messageContent);
        messageDiv.appendChild(messageTime);

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        if (save) {
            saveToHistory(text, sender, messageTime.textContent);
        }

        // Show notification if chat is minimized
        if (!chatContainer.classList.contains('active')) {
            document.querySelector('.notification-badge').style.display = 'flex';
        }
    }

    // Load chat history when page loads
    loadChatHistory();

    function showTypingIndicator() {
        if (!isTyping) {
            isTyping = true;
            typingIndicator.style.display = 'flex';
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    function hideTypingIndicator() {
        if (isTyping) {
            isTyping = false;
            typingIndicator.style.display = 'none';
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    function getFormattedTime() {
        const now = new Date();
        return now.toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: 'numeric',
            hour12: true 
        });
    }

    // Add this to your existing JavaScript
    function clearChatHistory() {
        localStorage.removeItem('chatHistory');
        chatMessages.innerHTML = ''; // Clear messages
        // Add back the welcome message
        addMessage(`Hello! I'm your AI career assistant. I can help you with:
            <ul>
                <li>Resume analysis and skill assessment</li>
                <li>Job recommendations and market insights</li>
                <li>Career guidance and industry trends</li>
            </ul>
            How can I assist you today?`, 'bot', 'Just now', true);
    }

    // Add a clear button to the chat header
    const clearButton = document.createElement('button');
    clearButton.className = 'clear-btn';
    clearButton.innerHTML = '<i class="fas fa-trash"></i>';
    clearButton.onclick = clearChatHistory;
    document.querySelector('.chat-header').appendChild(clearButton);
}); 