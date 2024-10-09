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
