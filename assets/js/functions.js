// Contact form — AJAX submission via Formspree
(function () {
    var form = document.getElementById('contact-form');
    if (!form) return;

    var submitBtn  = document.getElementById('contact-submit');
    var successMsg = document.getElementById('form-success');
    var errorMsg   = document.getElementById('form-error');

    form.addEventListener('submit', function (e) {
        e.preventDefault();

        submitBtn.disabled = true;
        submitBtn.textContent = 'Sending\u2026';
        successMsg.style.display = 'none';
        errorMsg.style.display   = 'none';

        fetch(form.action, {
            method:  'POST',
            body:    new FormData(form),
            headers: { 'Accept': 'application/json' }
        })
        .then(function (response) {
            if (response.ok) {
                form.reset();
                successMsg.style.display = 'block';
            } else {
                errorMsg.style.display = 'block';
            }
            submitBtn.disabled = false;
            submitBtn.textContent = 'Send Message';
        })
        .catch(function () {
            errorMsg.style.display = 'block';
            submitBtn.disabled = false;
            submitBtn.textContent = 'Send Message';
        });
    });
}());

document.addEventListener('DOMContentLoaded', function () {
    const toggleSwitch = document.querySelector('.switch input[type="checkbox"]');
    const regularTimeline = document.querySelector('.timeline:not(.fun-timeline)');
    const funTimeline = document.querySelector('.fun-timeline');

    toggleSwitch.addEventListener('change', function () {
        if (this.checked) {
            // Hide regular timeline and show fun timeline
            regularTimeline.style.display = 'none';
            funTimeline.style.display = 'block';
        } else {
            // Show regular timeline and hide fun timeline
            regularTimeline.style.display = 'block';
            funTimeline.style.display = 'none';
        }
    });
});
