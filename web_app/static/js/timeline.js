document.addEventListener('DOMContentLoaded', function() {

    // 1. Timeline Animation on Scroll
    const timelineItems = document.querySelectorAll('.timeline-item');

    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animation = `fadeInUp 0.5s ease-out forwards`;
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });

    timelineItems.forEach(item => {
        item.style.opacity = '0'; // Start with items invisible
        observer.observe(item);
    });

    // Add keyframes for the animation
    const styleSheet = document.createElement("style");
    styleSheet.type = "text/css";
    styleSheet.innerText = `
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    `;
    document.head.appendChild(styleSheet);


    // 2. Contact Form Success Animation
    const contactForm = document.getElementById('contactForm');
    const submitButton = contactForm.querySelector('button[type="submit"]');

    contactForm.addEventListener('submit', function(e) {
        e.preventDefault();

        // Change button to show success
        const originalButtonText = submitButton.innerHTML;
        submitButton.innerHTML = `<i class="fas fa-check"></i> Terkirim!`;
        submitButton.classList.remove('btn-primary');
        submitButton.classList.add('btn-success');
        submitButton.disabled = true;

        // Reset form and button after a few seconds
        setTimeout(() => {
            submitButton.innerHTML = originalButtonText;
            submitButton.classList.remove('btn-success');
            submitButton.classList.add('btn-primary');
            submitButton.disabled = false;
            contactForm.reset();
        }, 3000);
    });

});
