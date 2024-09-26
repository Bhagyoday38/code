import java.util.Arrays;
import java.util.Scanner;

public class AnagramChecker {

    public static void main(String[] args) {
       
        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter the first string:");
        String firstString = scanner.nextLine();
        System.out.println("Enter the second string:");
        String secondString = scanner.nextLine();

        
        boolean areAnagrams = areAnagrams(firstString, secondString);

        
        if (areAnagrams) {
            System.out.println("The strings are anagrams.");
        } else {
            System.out.println("The strings are not anagrams.");
        }
    }

    private static boolean areAnagrams(String firstString, String secondString) {
        
        firstString = firstString.toLowerCase();
        secondString = secondString.toLowerCase();
        char[] firstStringChars = firstString.toCharArray();
        char[] secondStringChars = secondString.toCharArray();
        Arrays.sort(firstStringChars);
        Arrays.sort(secondStringChars);

        
        return Arrays.equals(firstStringChars, secondStringChars);
    }
}
