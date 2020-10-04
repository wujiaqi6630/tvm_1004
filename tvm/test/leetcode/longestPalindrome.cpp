#include <iostream>
#include <string>
using namespace std;


bool isPalindrome(string s) {
	for (int i = 0; i < s.length(); ++i) {
		if (s[i] != s[s.length() - 1 - i])
			return false;
	}
	return true;
}
string longestPalindrome(string s) {
	int len = 1;
        string str1 = "b";
	string str =s[0] + "";
	for (int i = 0; i < s.length() - 1; ++i) {
		for (int j = 1; j < s.length() - i; ++j) {
			//C++没有substring,只有substr(start,len)
			if (isPalindrome(s.substr(i, j))) {
				//len = j > len ? j : len;
				str = j > len ? s.substr(i, j) : str;
                                len = j > len ? j : len;
			}
		}
	}
	return str;
}


int main() {
	string s = "abacdfgdcaba";
	string str = "";
	str = longestPalindrome(s);
	cout << "Longest string is : " << endl;
	cout << str << endl;
	//system("pause");
	return 0;
}
