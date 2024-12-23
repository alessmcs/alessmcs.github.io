// List of all navheads
const nav_menu = document.getElementsByClassName('navhead')
const sections = document.getElementsByClassName('partie')

function show_section(nav){
    const id = '#' + nav

    // Display section
    for (const section of sections) {
        if (section.id !== id) {
            section.classList.add('hidden')
        }
    }
    document.getElementById(id).classList.remove('hidden')
    document.getElementById(id).scrollTop = 0;

    // Set active navhead
    let nav_item = document.getElementById(nav)
    nav_item.classList.add('active')
        for (const i of nav_menu) {
            if ((i.id) !== nav_item.id){
                i.classList.remove('active')
            }
        }
}

function show_abstract(lang){
    if(lang === 'fr'){
        document.getElementById('abstract_fr').classList.remove('hidden')
        document.getElementById('abstract_eng').classList.add('hidden')
        document.getElementById('fr').classList.add('active')
        document.getElementById('eng').classList.remove('active')
    }
    else{
        document.getElementById('abstract_eng').classList.remove('hidden')
        document.getElementById('abstract_fr').classList.add('hidden')
        document.getElementById('eng').classList.add('active')
        document.getElementById('fr').classList.remove('active')
    }
}

window.onscroll = function() {scrollFunction2()};

function scrollFunction2() {
    const scrollTop = document.documentElement.scrollTop;
    const maxScroll = 300; // Define a maximum scroll distance for scaling
    const scale = Math.min(scrollTop / maxScroll, 1); // Scale value between 0 and 1

    // Scale font size proportionally
    const titleFontSize = 130 - (30 * scale); // Scale between 130% and 100%
    document.getElementById("project_title").style.fontSize = titleFontSize + "%";

    // Scale opacity proportionally
    document.getElementById("project_info").style.opacity = 1 - scale;

    // Scale maxHeight proportionally
    const infoMaxHeight = (1 - scale) * document.documentElement.scrollHeight; // Scale between full height and 0
    document.getElementById("project_info").style.maxHeight = infoMaxHeight + "px";
}