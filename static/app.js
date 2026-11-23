const answer = document.querySelector("#answer")
const answer_card = document.querySelector("#answer_card")
const form = document.querySelector("#form")
const input = document.querySelector("#input")
const submit_button = document.querySelector("#submit_button")
const welcome = document.querySelector("#welcome")
const examples = document.querySelectorAll(".example")

// 点击示例问题 = 直接发送：填入输入框后触发表单提交
for (const btn of examples) {
    btn.addEventListener("click", function(){
        input.value = btn.textContent
        form.requestSubmit()
    })
}

// 回答里的 markdown 符号变真样式：先转义 HTML 特殊字符（防回答内容被浏览器当标签），再把 **加粗** 和行首 ## 标题替换成加粗；最后把 [数字] 角标替换成上标；表格的竖线不做，要完美渲染得引库
function md_to_html(text){
    const safe = text.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
    return safe
        .replace(/^#{1,6} ?(.+)$/gm, "<strong>$1</strong>")
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\[(\d+)\]/g, '<sup class="cite">[$1]</sup>')
}

function render(c){
    if (c.evidence===null) {
        const louse = document.createElement('div')
        louse.className = "card"
        louse.textContent = "调用失败,未获得证据"
        answer_card.append(louse)
        return answer_card
    } else if(c.evidence.length===0) {
        const tool_reason = document.createElement('div')
        tool_reason.className = "card"
        if (c["name"] ==="calculate") {
            tool_reason.textContent = "计算参数出现问题"
        } else if (c["name"] === "web_search"){
            tool_reason.textContent = "没有搜到结果"
        } else {
            tool_reason.textContent = "未带回证据"
        }
        answer_card.append(tool_reason)
        return answer_card
    } else {
        for (let index = 0; index < c.evidence.length; index++) {
            const element = c.evidence[index];
            const card = document.createElement("div")
            if (element["来源"]==="检索"){
                card.className = "card"
                card.textContent = `${element["文件名"]},RRF分 ${element["RRF分"].toFixed(5)}`
            } else if(element["来源"]==="搜索") {
                const a = document.createElement('a')
                a.href = element["URL"]
                card.className = "card"
                card.textContent = `${element["标题"]},${element["内容"]}`
                a.textContent = a.href
                card.append(a)
            }else{
                card.className = "card"
                card.textContent = `${element["表达式"]},${element["结果"]}`
            }
            answer_card.append(card)
        }
        
    }
}

// 渲染引用清单：只列正文里实际出现角标的来源，不把检索到但没采用的块列出来
function render_citations(answer_text, citations){
    let box = document.querySelector("#citation_box")
    if (!box) {
        box = document.createElement("div")
        box.id = "citation_box"
        answer.after(box)
    }
    box.textContent = ""
    // 从原始回答文本提取所有角标编号并去重（同一编号可能出现多次）
    const used = [...answer_text.matchAll(/\[(\d+)\]/g)].map(m => m[1])
    const used_unique = [...new Set(used)]
    if (used_unique.length === 0 || !citations) {
        box.remove()   // 拒答/计算类没有角标，不显示引用框
        return
    }
    for (const num of used_unique) {
        const src = citations[num]
        if (!src) continue   // 编号在映射里查不到（理论上不该发生），跳过不报错
        const row = document.createElement("div")
        row.className = "citation_item"
        row.textContent = `[${num}] ${src["文件名"]} · ${src["小节"]}`
        box.append(row)
    }
}

form.addEventListener('submit',async function(e){
    e.preventDefault()
    // 首次提问后欢迎块消失
    if (welcome.parentNode) {
        welcome.remove()
    }
    answer.textContent = "思考中..."
    answer_card.textContent = ""
    submit_button.disabled = true
    try {
        const response = await fetch("/ask",{
        method:"POST",
        headers: {"Content-Type": "application/json"},
        body:JSON.stringify(
            {mes:input.value}
        )
        })
        const data =await response.json()
        if (response.ok) {
            answer.innerHTML = md_to_html(data["回答"])
            render_citations(data["回答"], data["引用来源"])
            for (const rec of data["路由"]) render(rec)
            input.value = ''
        } else {
            answer.textContent = data.detail
        }
    } catch (error) {
        console.error(error)
        answer.textContent = "服务未连接"
    }
    finally {
        submit_button.disabled = false
    }
})