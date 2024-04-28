function sendMessage() {
    var userInput = document.getElementById("user-input").value;
    var chatContainer = document.getElementById("chat-container");
    var messageElement = document.createElement("div");
    messageElement.textContent = "You: " + userInput;
    chatContainer.appendChild(messageElement);
    document.getElementById("user-input").value = "";
    // Send user input to the server for processing (using AJAX or Fetch)
    // Receive response from server and append to chat container
}
