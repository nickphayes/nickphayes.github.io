
// making arrow fade in and out on scroll

$(window).scroll(function() {
    if ($(this).scrollTop()> 5) {
        $('.arrow').fadeOut();
     }
    else {
      $('.arrow').fadeIn();
     }
 });

// setting smooth scrolling on click event
// takes in target item as selection
function scrollWindow(selection){
    document.querySelector(selection).scrollIntoView({ 
        behavior: 'smooth',
        block: 'center'
      });
}

// typewriter function for title page
var i = 0 ; 
var speed = 100; // typing speed in ms
var txt = 'Nicholas Hayes'; 
var target = document.getElementById('homename'); 

function typeWriterTitle() {
  if (i < txt.length){
    target.innerHTML += txt.charAt(i);
    i ++;
    setTimeout(typeWriterTitle, 100);
  }
  else {
    setTimeout(moveDiv, 1000);
  }
}

//moving title div function
//triggered by previous function call
//using target variable previously defined in above function

function moveDiv() {
  target.style.top = '0'; 
  target.style.left = '0'; 
  target.style.transform = 'translate(0,0)'; 
  target.style.marginTop = '7%'; 
  target.style.marginLeft = '1em'; 
}