function langDetect() {
	var myHeaders = new Headers();
	myHeaders.append('Content-Type', 'application/json');
	let data = document.querySelector('#lang_data').value;

	if (data === '') {
		return;
	}

	var raw = JSON.stringify({
		lang_data: data,
	});

	var requestOptions = {
		method: 'POST',
		headers: myHeaders,
		body: raw,
		redirect: 'follow',
	};

	fetch('http://localhost:8000/lang_detect/', requestOptions)
		.then((response) => response.text())
		.then((result) => {
			document.querySelector('.result').innerHTML = result;
			let resultId = document.querySelector('.result');
			resultId.style.display = 'block';
		})
		.catch((error) => console.log('error', error));
}
