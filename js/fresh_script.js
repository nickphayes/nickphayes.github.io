// responsive hamburger menu

const hamburger = document.querySelector(".hamburger");
const navMenu = document.querySelector(".nav-menu");

hamburger.addEventListener("click", mobileMenu);

function mobileMenu() {
    hamburger.classList.toggle("active");
    navMenu.classList.toggle("active");
}

const navLink = document.querySelectorAll(".nav-link");

navLink.forEach(n => n.addEventListener("click", closeMenu));

function closeMenu() {
    hamburger.classList.remove("active");
    navMenu.classList.remove("active");
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
    setTimeout(moveDiv, 750);
  }
}

//moving title div function
//triggered by previous function call
//using target variable previously defined in above function

var targetDiv = document.getElementById('nametitle');

function moveDiv() {
    targetDiv.style.top = '1em'; 
    targetDiv.style.left = '1em'; 
    targetDiv.style.transform = 'translate(0,0)'; 
    target.style.fontSize = '12px';
    target.style.letterSpacing = '3px';
    setTimeout(fadeBackground, 0)
}

//function for fading background on home screen
//triggered by previous function call

var fader = document.getElementById('fade-background');

function fadeBackground() {
    fader.style.opacity = '0';
    setTimeout(function(){
        fader.style.zIndex = '-1';
    }, 2000)
}