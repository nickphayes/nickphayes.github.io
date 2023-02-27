
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

// typewriter function
var i = 0 ; 
function typeWriter(text, moveToTopLeft) {
    if (i < text.length) {
      document.getElementById("homename").textContent += text.charAt(i);
      i++;
      setTimeout(function() {
        typeWriter(text, moveToTopLeft)
      }, 100);
    } else if (typeof moveToTopLeft == "function") {
      setTimeout(moveToTopLeft, 0);
    }
  }
  
// moving title container from center to top left

function moveToTopLeft() {
    var div = document.querySelector(".title a");
    div.style.position = "absolute";
    div.style.left = "0";
    div.style.top = "0";
  }
  
//STOPPING POINT: FIXING MOVING HOMENAME



