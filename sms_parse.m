function [ words ] = sms_parse( msg )
%This file Parses an sms message returns array of words
% Input: msg string containing sms message
% Output: words   cell array of words in message 

% convert GBP symbol to dollar sign and add a space
msg = regexprep(msg,'[£$]',' $ ');

% convert :// to special token
msg = regexprep(msg, '\://', ' __url__ ');

% convert non-alaphnumeric to space
msg = lower(regexprep(msg,'[^_\s$a-zA-Z0-9]',' '));

% split into words
C = textscan(msg, '%s');
words = C{1};

end
