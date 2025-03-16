function openModal(type) {
    const modal = document.getElementById(`${type}Modal`);
    modal.style.display = 'block';
}

function closeModal(type) {
    const modal = document.getElementById(`${type}Modal`);
    modal.style.display = 'none';
}

// Close modal when clicking outside
window.onclick = function(event) {
    const loginModal = document.getElementById('loginModal');
    const registerModal = document.getElementById('registerModal');
    
    if (event.target === loginModal) {
        loginModal.style.display = 'none';
    }
    if (event.target === registerModal) {
        registerModal.style.display = 'none';
    }
}

// Remove the form submission handlers since we're using server-side handling 