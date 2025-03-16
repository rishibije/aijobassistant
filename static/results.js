document.addEventListener('DOMContentLoaded', function() {
    // Handle skill tag removal
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('remove')) {
            e.target.parentElement.remove();
        }
    });
});