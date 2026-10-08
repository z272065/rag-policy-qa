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
                card.textContent = `${element["文件名"]},相似度 ${element["相似度"]}`
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
            answer.textContent = data["回答"]
            for (const rec of data["路由"]) render(rec)
            input.value = ''
        } else {
            answer.textContent = data.detail
        }
    } catch (error) {
        answer.textContent = "服务未连接"
    }
    finally {
        submit_button.disabled = false
    }
})