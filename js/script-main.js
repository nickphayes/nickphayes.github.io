
// making arrow fade in and out

$(window).scroll(function() {
    if ($(this).scrollTop()> 5) {
        $('.arrow').fadeOut();
     }
    else {
      $('.arrow').fadeIn();
     }
 });

// setting smooth scrolling on click event
function scrollWindow(selection){
    document.querySelector(selection).scrollIntoView({ 
        behavior: 'smooth',
        block: 'center'
      });
}




