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
    setTimeout(moveDiv, 1000);
  }
}

//moving title div function
//triggered by previous function call
//using target variable previously defined in above function

var targetDiv = document.getElementById('nametitle');

function moveDiv() {
    // stopping point: gradually moving title name to top left
    //not sure why, but transition attribute in CSS not working
    // might have to play with child div to get to work
    // or just change container flexbox
}