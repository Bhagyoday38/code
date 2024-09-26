import java.util.Scanner;

public class RemoveCharacter
 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter the size of the array: ");
        int n = scanner.nextInt();

        
        int[] arr = new int[n];
        System.out.println("Enter the elements of the array: ");
        for (int i = 0; i < n; i++) {
            arr[i] = scanner.nextInt();
        } 
        int smallest = arr[0];
        int largest = arr[0];
        for (int i = 1; i < n; i++) 
{
            if (arr[i] < smallest) 
{
                smallest = arr[i];
            }
            if (arr[i] > largest)
 {
                largest = arr[i];
            }
        }

        
        System.out.println("The smallest element is: " + smallest);
        System.out.println("The largest element is: " + largest);
    }
}

