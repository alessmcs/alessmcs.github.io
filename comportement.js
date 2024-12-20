// List of all navheads
const nav_menu = document.getElementsByClassName('navhead')
const sections = document.getElementsByClassName('partie')

document.addEventListener('DOMContentLoaded', function() {
    console.log('loaded')
    for (const p of sections){
        console.log(p.id)
    }

    for (const nav_item of nav_menu) { // Loop through each item in the HTMLCollection
        nav_item.addEventListener('click', function() {
            let ref = nav_item.querySelector('a').getAttribute('href')

            console.log(ref)
            console.log(nav_item.id)

            if (localStorage.getItem('active') !== ref) {
                document.getElementById(ref).classList.remove('hidden')
                document.getElementById(ref).scrollTop = 0
                for (const section of sections) {
                    if (section.id !== ref) {
                        section.classList.add('hidden')
                    }
                }

                nav_item.classList.add('active')

                for (const i of nav_menu) {
                    if ((i.id) !== nav_item.id){
                        i.classList.remove('active')
                    }
                }
            }

            localStorage.setItem('active', ref)
        });
    }
})
